"""Agent episodes: the turn loop, grading rule, held-out split and the sandbox tools."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

from training_data import pr_task_agent as agent

RUNNER = Path(__file__).parents[1] / "data" / "pr_task_runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("pr_task_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(name, arguments="{}"):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


class EpisodeTests(unittest.TestCase):
    def test_loop_runs_tools_until_submit_or_limits(self):
        scripts = {
            "submits": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call("bash", '{"command": "ls"}')],
                },
                {"role": "assistant", "content": "done", "tool_calls": [call("submit")]},
            ],
            "answers": [{"role": "assistant", "content": "I cannot."}],
            "loops": [{"role": "assistant", "content": "", "tool_calls": [call("bash", "{bad")]}]
            * 5,
        }
        episodes = [{"name": n, "messages": [{"role": "user", "content": n}]} for n in scripts]
        episodes.append({"name": "long", "messages": [{"role": "user", "content": "long"}]})

        def policy(conversations):
            replies = []
            for messages in conversations:
                name = messages[0]["content"]
                turn = sum(m["role"] == "assistant" for m in messages)
                replies.append(None if name == "long" else dict(scripts[name][turn]))
            return replies

        executed = []

        def execute(episode, name, arguments):
            executed.append((episode["name"], name, arguments))
            return "ok"

        agent.run_episodes(episodes, policy, execute, max_turns=3, workers=2)
        ends = {e["name"]: (e["end"], e["turns"]) for e in episodes}
        self.assertEqual(
            ends,
            {
                "submits": ("submitted", 2),
                "answers": ("answered", 1),
                "loops": ("max_turns", 3),
                "long": ("context", 0),
            },
        )
        self.assertIn(("submits", "bash", {"command": "ls"}), executed)
        self.assertIn(("loops", "bash", {}), executed)
        submitted = episodes[0]["messages"]
        self.assertEqual(
            [m["role"] for m in submitted], ["user", "assistant", "tool", "assistant", "tool"]
        )
        self.assertEqual(submitted[-1]["content"], "submitted")

    def test_resolved_needs_every_listed_test_to_pass(self):
        verification = {"fail_to_pass": ["a"], "pass_to_pass": ["b"]}
        graded = {"stage": "graded", "results": {"a": "pass", "b": "pass"}}
        self.assertTrue(agent.resolved(graded, verification))
        graded["results"]["b"] = "fail"
        self.assertFalse(agent.resolved(graded, verification))
        self.assertFalse(agent.resolved({"stage": "patch_failed"}, verification))

    def test_heldout_keeps_one_owner_on_one_side(self):
        owners = [f"owner{i}" for i in range(200)]
        flags = [agent.heldout(f"{o}/a") for o in owners]
        self.assertEqual(flags, [agent.heldout(f"{o}/b") for o in owners])
        self.assertTrue(0.1 < sum(flags) / len(flags) < 0.4)

    def test_training_row_keeps_valid_json_arguments(self):
        messages = [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", **call("bash", "{bad")}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "out"},
        ]
        row = agent.training_row(messages)
        self.assertEqual(row["messages"][1]["tool_calls"][0]["function"]["arguments"], "{}")
        self.assertEqual(row["messages"][1]["content"], "")
        self.assertNotIn("tool_call_id", row["messages"][2])


class RunnerToolTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()
        self.folder = tempfile.TemporaryDirectory()
        repo = Path(self.folder.name, "repo")
        repo.mkdir()
        self.runner.REPO = repo

        def git(*args):
            return subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        git("init", "-q")
        (repo / "src.py").write_text("def f():\n    return 1\n")
        (repo / "tests").mkdir()
        (repo / "tests/test_src.py").write_text("from src import f\n")
        git("add", "-A")
        git("commit", "-q", "-m", "parent")
        self.parent = git("rev-parse", "HEAD")
        (repo / "src.py").write_text("def f():\n    return 2\n")
        (repo / "tests/test_src.py").write_text("from src import f\nassert f() == 2\n")
        (repo / "tests/test_new.py").write_text("assert True\n")
        git("add", "-A")
        git("commit", "-q", "-m", "fix f")
        self.sha = git("rev-parse", "HEAD")
        self.repo = repo

    def tearDown(self):
        self.folder.cleanup()

    def test_workspace_hides_the_commit_and_tools_edit_and_diff(self):
        task = {
            "sha": self.sha,
            "parent": self.parent,
            "source_paths": ["src.py"],
            "test_paths": ["tests/test_src.py", "tests/test_new.py"],
        }
        self.assertEqual(self.runner.workspace(task)["stage"], "ready")
        self.assertIn("return 1", (self.repo / "src.py").read_text())
        self.assertNotIn("assert", (self.repo / "tests/test_src.py").read_text())
        self.assertFalse((self.repo / "tests/test_new.py").exists())
        log = self.runner.tool("bash", {"command": "git log --oneline | wc -l"})
        self.assertEqual(log, "exit code: 0\n1\n")
        self.assertIn(
            "edited", self.runner.tool("edit_file", {"path": "src.py", "old": "1", "new": "2"})
        )
        self.assertIn(
            "occurs 0 times",
            self.runner.tool("edit_file", {"path": "src.py", "old": "9", "new": "8"}),
        )
        self.assertIn("    1  def f():", self.runner.tool("read_file", {"path": "src.py"}))
        self.assertIn("error", self.runner.tool("read_file", {"path": "../outside"}))
        diff = self.runner.patch()["patch"]
        self.assertIn("-    return 1\n+    return 2", diff)


if __name__ == "__main__":
    unittest.main()
