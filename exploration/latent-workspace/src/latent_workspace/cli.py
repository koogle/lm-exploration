import argparse
import json

from .config import Config


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--tokens", type=int, default=32)
    gen.add_argument("--temperature", type=float, default=0)
    draft = sub.add_parser("draft", help="DFlash-style greedy drafting with Qwen verification")
    draft.add_argument("--checkpoint", required=True)
    draft.add_argument("--prompt", required=True)
    draft.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    if args.command == "train":
        from .train import run

        print(json.dumps(run(Config.load(args.config), args.output, resume=args.resume), indent=2))
    elif args.command == "draft":
        from .dflash_cli import generate_checkpoint

        print(json.dumps(generate_checkpoint(args.checkpoint, args.prompt, args.tokens), indent=2))
    else:
        import torch

        from .backbone import FrozenBackbone
        from .evaluate import generate
        from .models import BlockHead

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = Config(**state["config"]).validate()
        cfg.model_revision = state["model_revision"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone = FrozenBackbone(cfg, device)
        backbone.jacobians = state.get("jacobians")
        head = BlockHead(backbone.hidden_size, cfg, state["variant"]).to(device)
        head.load_state_dict(state["head"])
        prefix = backbone.tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
        if not prefix.numel() or args.tokens <= 0:
            raise ValueError("prompt must be nonempty and tokens positive")
        ids = generate(head, backbone, prefix, args.tokens, args.temperature)
        print(backbone.tokenizer.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
