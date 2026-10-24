import unittest
import unittest.mock

import helpers  # noqa: F401  (sets sys.path for the project modules below)

# isort: split


class AdoptionTest(unittest.TestCase):
    def test_restart_adopts_only_its_own_unfailed_calls(self):
        import modal_app

        previous = {
            "supervisor_call_id": "fc-1",
            "calls": {"a": {"id": "fc-a"}, "b": {"id": "fc-b", "error": "boom"}},
        }
        self.assertEqual(modal_app.adoptable(previous, "fc-1"), {"a": {"id": "fc-a"}})
        self.assertEqual(modal_app.adoptable(previous, "fc-2"), {})
        self.assertEqual(modal_app.adoptable({}, "fc-1"), {})


class PollTest(unittest.TestCase):
    def test_unfinished_call_is_pending_not_failed(self):
        import modal
        import modal_app

        class Call:
            def __init__(self, outcome):
                self.outcome = outcome

            def get(self, timeout=None):
                if isinstance(self.outcome, BaseException):
                    raise self.outcome
                return self.outcome

        self.assertEqual(modal_app.poll(Call(TimeoutError())), ("pending", None))
        self.assertEqual(
            modal_app.poll(Call(modal.exception.TimeoutError())), ("pending", None)
        )
        self.assertEqual(modal_app.poll(Call({"ok": 1})), ("done", {"ok": 1}))
        state, _ = modal_app.poll(
            Call(modal.exception.FunctionTimeoutError("child timed out"))
        )
        self.assertEqual(state, "error")
        state, _ = modal_app.poll(Call(modal.exception.InputCancellation("cancelled")))
        self.assertEqual(state, "error")
        state, _ = modal_app.poll(Call(ValueError("boom")))
        self.assertEqual(state, "error")


if __name__ == "__main__":
    unittest.main()
