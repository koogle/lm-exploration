import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B"
    model_revision: str = "main"
    allow_compatible_backbone_change: bool = False
    normalize_token_inputs: bool = False
    position_readout: bool = False
    residual_readout: bool = False
    layer_dropout: float = 0.0
    null_attention: bool = False
    study_layer_controls: bool = False
    study_train_diagnostics: int = 96
    study_scan_only: bool = False
    scan_attention: bool = False
    study_eval_only: bool = False
    study_eval_parents: tuple[str, ...] = ()
    study_eval_hashes: dict[str, str] = field(default_factory=dict)
    study_resume: str = ""
    study_reference: str = ""
    study_parent: str = ""
    study_sizes: tuple[int, ...] = ()
    study_checkpoints: tuple[int, ...] = (100, 500, 1000, 2000)
    evaluate_test: bool = False
    scan_steps: int = 0
    select_layers_by_probe: bool = False
    selected_layer_count: int = 3
    corpus: str = "wikitext"
    test_examples: int = 384
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    dataset_revision: str = "main"
    layers: tuple[int, ...] = (6, 12, 18, 24)
    jspace_prompts: int = 8
    jspace_top_k: int = 25
    jspace_dim_batch: int = 16
    selected_layers: tuple[int, ...] = ()
    block_size: int = 4
    context_length: int = 128
    memory_tokens: int = 16
    slots: int = 8
    width: int = 512
    heads: int = 8
    decoder_layers: int = 4
    refinement_rounds: int = 2
    diffusion_steps: int = 4
    batch_size: int = 8
    train_examples: int = 8192
    eval_examples: int = 256
    steps: int = 1000
    eval_every: int = 250
    learning_rate: float = 0.0003
    seed: int = 42
    max_seconds: int = 1800
    variants: tuple[str, ...] = (
        "parallel:final",
        "diffusion:final",
        "diffusion:layer12",
        "diffusion:multi",
        "diffusion:attention12",
        "diffusion:attention_multi",
        "diffusion:jspace",
    )

    def validate(self):
        if not 0 <= self.layer_dropout <= 1 or self.study_train_diagnostics < 3:
            raise ValueError("invalid layer dropout or training diagnostic count")
        if self.study_layer_controls and (self.study_resume or self.evaluate_test):
            raise ValueError("layer control suite cannot resume or evaluate test")
        if self.study_sizes and (
            any(n <= 0 or n % 3 or n > self.train_examples for n in self.study_sizes)
            or not self.study_checkpoints
            or min(self.study_checkpoints) <= 0
        ):
            raise ValueError("invalid learning-curve sizes or checkpoints")
        if self.corpus not in ("wikitext", "mixed"):
            raise ValueError("unknown corpus")
        if self.corpus == "mixed" and any(
            x < 3 or x % 3 for x in (self.train_examples, self.eval_examples, self.test_examples)
        ):
            raise ValueError("mixed split window counts must be positive multiples of three")
        for key in (
            "jspace_prompts",
            "jspace_top_k",
            "jspace_dim_batch",
            "block_size",
            "context_length",
            "memory_tokens",
            "slots",
            "width",
            "heads",
            "decoder_layers",
            "refinement_rounds",
            "diffusion_steps",
            "batch_size",
            "train_examples",
            "eval_examples",
            "steps",
            "eval_every",
            "max_seconds",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.width % self.heads or self.memory_tokens > self.context_length:
            raise ValueError("width must divide heads; memory must fit context")
        if tuple(sorted(self.layers)) != tuple(self.layers):
            raise ValueError("layers must be in increasing order")
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 1:
            raise ValueError("layers must be unique positive hidden-state indices")
        if not set(self.selected_layers).issubset(self.layers):
            raise ValueError("selected_layers must be extracted layers")
        if len(self.layers) < 2 and any(v.endswith(":jspace") for v in self.variants):
            raise ValueError("J-space requires an intermediate and final layer")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be nonempty and unique")
        if not 0 < self.learning_rate < 1:
            raise ValueError("invalid learning rate")
        for variant in self.variants:
            kind, condition = variant.split(":")
            if kind == "probe":
                if int(condition.removeprefix("attention")) not in self.layers:
                    raise ValueError("probe layer absent from extraction layers")
            elif kind not in ("parallel", "diffusion") or condition not in (
                "final",
                "multi",
                "all",
                "attention_all",
                "best",
                "attention_multi",
                "jspace",
                *[f"layer{i}" for i in self.layers],
                *[f"attention{i}" for i in self.layers],
            ):
                raise ValueError(f"unknown variant: {variant}")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        for key in (
            "layers",
            "selected_layers",
            "variants",
            "study_sizes",
            "study_checkpoints",
            "study_eval_parents",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values).validate()
