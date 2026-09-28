#!/usr/bin/env python3
"""NEXUS-X's single user entry point (CLI or interactive control center)."""
from __future__ import annotations
import argparse
import json
import sys

from nexus.config import Config
from nexus.orchestrator import Orchestrator

MENU = """
============================================
        NEXUS-X CONTROL CENTER — STAGE 1
============================================
  [1] CREATE AND TRAIN
  [2] FINE-TUNE
  [3] USE / INFERENCE
  [4] BENCHMARK
  [5] ARCHITECTURE SEARCH — later stage
  [6] STATUS
  [0] EXIT
============================================
"""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experimental CPU-first NEXUS-X")
    p.add_argument("command", nargs="?", choices=["create", "finetune", "use", "bench", "search", "status"])
    p.add_argument("--config", default="nexus_config.yaml")
    p.add_argument("--model", help="Existing .nexus model directory")
    p.add_argument("--output", help="New .nexus directory (or JSON report for bench)")
    p.add_argument("--resume", action="store_true", help="Resume --model including optimizer and sampler")
    p.add_argument("--max-steps", type=int, help="Stop after N optimizer updates; preserves full schedule")
    p.add_argument("--prompt", help="One-shot generation; otherwise interactive")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--baseline", action="store_true", help="Train optional torch Transformer control")
    return p


def use_loop(app: Orchestrator) -> None:
    print("Independent prompts (no conversation memory). /exit to return.")
    while True:
        try:
            prompt = input("You> ")
        except EOFError:
            return
        if prompt.strip() == "/exit":
            return
        print("NEXUS>", app.use(prompt)["text"])


def interactive(app: Orchestrator) -> None:
    while True:
        print(MENU)
        try:
            choice = input("> ").strip()
            if choice == "0":
                return
            if choice == "1":
                app.create()
            elif choice == "2":
                source = input("Base model [config]: ").strip() or None
                output = input("New output .nexus [config]: ").strip() or None
                app.finetune(source, output)
            elif choice == "3":
                source = input("Model [model.nexus]: ").strip() or "model.nexus"
                app.load(source)
                use_loop(app)
            elif choice == "4":
                source = input("Model [model.nexus]: ").strip() or "model.nexus"
                app.bench(source, "runs/benchmark.json")
            elif choice == "5":
                print("Architecture search is outside stage 1; no search results are claimed.")
            elif choice == "6":
                print(json.dumps(app.status(), indent=2, ensure_ascii=False))
            else:
                print("Choose 0–6.")
        except EOFError:
            return
        except (ValueError, OSError, RuntimeError, FloatingPointError) as error:
            app.context.logger.error("[ERROR] %s", error)


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be positive")
    if args.resume and (args.command not in {"create", "finetune"} or not args.model):
        p.error("--resume requires create/finetune and --model")
    if args.command == "search":
        p.error("Architecture search is outside stage 1")
    try:
        app = Orchestrator(Config.load(args.config))
        if args.command is None:
            interactive(app)
        elif args.command == "create":
            app.create(args.output, args.model if args.resume else None, args.max_steps)
        elif args.command == "finetune":
            app.finetune(args.model, args.output, args.resume, args.max_steps)
        elif args.command == "use":
            app.load(args.model or "model.nexus")
            if args.prompt is None:
                use_loop(app)
            else:
                options = {k: v for k, v in {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}.items() if v is not None}
                print(json.dumps(app.use(args.prompt, **options), ensure_ascii=False, indent=2))
        elif args.command == "bench":
            result = app.bench(args.model or "model.nexus", args.output or "runs/benchmark.json", args.baseline)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "status":
            if args.model:
                app.load(args.model)
            print(json.dumps(app.status(), ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("\n[MODE] Interrupted. Last completed checkpoint is preserved.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, FloatingPointError, KeyError, TypeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
