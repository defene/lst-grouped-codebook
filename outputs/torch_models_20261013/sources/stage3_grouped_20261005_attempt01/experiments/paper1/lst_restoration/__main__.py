"""No implicit launch: help/validate use the standard library only."""
import argparse
import json
from pathlib import Path

from .config import validate, write_json, is_pretrained


def main():
    parser = argparse.ArgumentParser(description="LST discrete restoration framework")
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate", help="Static configuration checks only")
    validation.add_argument("--config", required=True)
    checks = commands.add_parser("preflight", help="Explicit FUTURE runtime tests; can execute forward/backward")
    checks.add_argument("--config", required=True)
    checks.add_argument("--output", required=True)
    checks.add_argument("--device", default="cpu")
    checks.add_argument("--part", choices=("components", "data-budget", "model", "all"), default="components")
    freeze = commands.add_parser("assemble-preflight", help="Merge existing checks, never invent passing gates")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--reports", nargs="+", required=True)
    freeze.add_argument("--output", required=True)
    alignment = commands.add_parser("alignment-audit", help="Compare measured parameters and rate for four core quantizers")
    alignment.add_argument("--reports", nargs="+", required=True)
    alignment.add_argument("--output", required=True)
    for name in ("train", "run", "evaluate"):
        command = commands.add_parser(name, help="Explicit execution; config execution guard applies")
        command.add_argument("--config", required=True)
        command.add_argument("--output", required=True)
        if name == "evaluate":
            command.add_argument("--checkpoint", required=True, help="Final checkpoint or literal pretrained")
    report = commands.add_parser("report")
    report.add_argument("--job", required=True)
    gallery = commands.add_parser("gallery", help="Render stored arrays, no inference")
    gallery.add_argument("--cell", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--a", required=True)
    comparison.add_argument("--b", required=True)
    comparison.add_argument("--output", required=True)
    stability = commands.add_parser("token-stability", help="Paired token agreement within one frozen model")
    stability.add_argument("--a", required=True)
    stability.add_argument("--b", required=True)
    stability.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if hasattr(args, "config") else None
    if args.command == "validate":
        result = validate(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            raise SystemExit(1)
        return
    if args.command == "preflight":
        directory = Path(args.output)
        if args.part in ("components", "all"):
            from .preflight import run_component_checks
            run_component_checks(config, directory / "components.json")
        if args.part in ("data-budget", "all"):
            from .audit_data_budget import run_data_budget_checks
            run_data_budget_checks(config, directory / "data_budget.json")
        if args.part in ("model", "all"):
            from .audit_model import run_model_checks
            run_model_checks(config, directory / "model.json", device=args.device)
        return
    if args.command == "assemble-preflight":
        from .runtime import protocol_digest, source_hashes
        gates, reports = {}, {}
        for path in args.reports:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if value.get("protocol_config_sha256") != protocol_digest(config) or value.get("source_hashes") != source_hashes():
                raise ValueError("Preflight evidence belongs to different code/config: " + path)
            if value.get("status") not in ("passed", "synthetic_model_checks_complete_other_gates_pending"):
                raise ValueError("Failed or unfinished preflight evidence: " + path)
            for key, passed in value.get("gates", {}).items():
                if passed is True:
                    gates[key] = True
            reports[path] = value
        required = ("components", "model_contract", "backward_finite", "parameter_rate_audit",
                    "budget_resume", "resource_profile", "data_contract", "grouped_mechanism")
        write_json(args.output, {"protocol_config_sha256": protocol_digest(config), "source_hashes": source_hashes(),
            "gates": {name: gates.get(name, False) for name in required}, "reports": reports,
            "passed": all(gates.get(name) is True for name in required), "authorizes_training": False})
        return
    if args.command == "alignment-audit":
        from .report import alignment_audit
        print(json.dumps(alignment_audit(args.reports, args.output), ensure_ascii=False, indent=2))
        return
    if args.command in ("train", "run", "evaluate"):
        from .runtime import train
        from .evaluate import evaluate
        from .tracking import session
        output = Path(args.output)
        with session(config, output):
            if args.command == "evaluate":
                evaluate(config, args.checkpoint, output)
            else:
                if args.command == "run" and is_pretrained(config):
                    evaluate(config, "pretrained", output / "pretrained")
                checkpoint = train(config, output)
                if args.command == "run":
                    evaluate(config, checkpoint, output / "final")
                    from .report import build_report
                    print(build_report(output))
        return
    from .report import build_report, compare_cells, render_gallery, token_stability
    if args.command == "report":
        print(build_report(args.job))
    elif args.command == "gallery":
        print(render_gallery(args.cell))
    elif args.command == "token-stability":
        print(json.dumps(token_stability(args.a, args.b, args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(compare_cells(args.a, args.b, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
