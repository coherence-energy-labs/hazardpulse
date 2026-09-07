"""Command-line entry points for HazardPulse utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tuple_from_repeat(values: list[str] | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return fallback
    return tuple(value.upper() for value in values)


def _ints_from_csv(text: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not text:
        return fallback
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _load_hurricane_support() -> dict | None:
    """Import optional hurricane CLI helpers when available."""
    try:
        from hazardpulse.hurricane.ablation import (
            default_feature_ablation_specs,
            run_operational_ri_ablation_suite,
            select_feature_names_for_spec,
        )
        from hazardpulse.hurricane.atcf import (
            DEFAULT_AID_MODELS,
            DEFAULT_LEAD_TIMES,
            build_operational_ri_dataset,
            summarize_operational_ri_cases,
            write_operational_ri_dataset,
        )
        from hazardpulse.hurricane.operational_ri import (
            benchmark_operational_ri,
            load_operational_ri_cases,
        )
        from hazardpulse.hurricane.tuning import (
            best_known_operational_ri_candidate,
            default_operational_ri_tuning_candidates,
            run_operational_ri_tuning_search,
        )
    except Exception:
        return None

    return {
        "DEFAULT_AID_MODELS": DEFAULT_AID_MODELS,
        "DEFAULT_LEAD_TIMES": DEFAULT_LEAD_TIMES,
        "build_operational_ri_dataset": build_operational_ri_dataset,
        "summarize_operational_ri_cases": summarize_operational_ri_cases,
        "write_operational_ri_dataset": write_operational_ri_dataset,
        "default_feature_ablation_specs": default_feature_ablation_specs,
        "run_operational_ri_ablation_suite": run_operational_ri_ablation_suite,
        "select_feature_names_for_spec": select_feature_names_for_spec,
        "benchmark_operational_ri": benchmark_operational_ri,
        "load_operational_ri_cases": load_operational_ri_cases,
        "best_known_operational_ri_candidate": best_known_operational_ri_candidate,
        "default_operational_ri_tuning_candidates": default_operational_ri_tuning_candidates,
        "run_operational_ri_tuning_search": run_operational_ri_tuning_search,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hazardpulse")
    subparsers = parser.add_subparsers(dest="command")

    earthquake_parser = subparsers.add_parser(
        "earthquake-v4",
        help="Run the regional earthquake model",
    )
    earthquake_parser.add_argument(
        "--honest",
        action="store_true",
        help="Run honest mode (M6+, 100 km, 90 days, 5:1 controls).",
    )
    earthquake_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory for saved results.",
    )
    earthquake_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )

    hurricane_support = _load_hurricane_support()
    if hurricane_support is None:
        return parser

    hurricane_parser = subparsers.add_parser("hurricane", help="Hurricane utilities")
    hurricane_sub = hurricane_parser.add_subparsers(dest="hurricane_command")

    dataset_parser = hurricane_sub.add_parser(
        "build-operational-ri-dataset",
        help="Build advisory-time RI cases from archived ATCF decks",
    )
    dataset_parser.add_argument("--start-year", type=int, required=True)
    dataset_parser.add_argument("--end-year", type=int, required=True)
    dataset_parser.add_argument("--basin", action="append", dest="basins")
    dataset_parser.add_argument("--aid", action="append", dest="aids")
    dataset_parser.add_argument("--lead-times", default=None)
    dataset_parser.add_argument("--limit-storms", type=int, default=None)
    dataset_parser.add_argument("--out", required=True)

    benchmark_parser = hurricane_sub.add_parser(
        "benchmark-operational-ri",
        help="Benchmark an advisory-time RI dataset against operational aids",
    )
    benchmark_parser.add_argument("--dataset", required=True)
    benchmark_parser.add_argument("--test-start-year", type=int, required=True)
    benchmark_parser.add_argument("--candidate", default="best_known")

    sweep_parser = hurricane_sub.add_parser(
        "sweep-operational-ri",
        help="Run feature-group ablations across rolling operational RI holdouts",
    )
    sweep_parser.add_argument("--dataset", required=True)
    sweep_parser.add_argument("--holdouts", default="2022,2023,2024")
    sweep_parser.add_argument(
        "--preset",
        choices=("quick", "full", "taucoh", "organism"),
        default="quick",
    )
    sweep_parser.add_argument("--only", action="append", dest="only_specs")
    sweep_parser.add_argument("--out", default=None)

    search_parser = hurricane_sub.add_parser(
        "search-operational-ri",
        help="Run a compact configuration search across rolling operational RI holdouts",
    )
    search_parser.add_argument("--dataset", required=True)
    search_parser.add_argument("--holdouts", default="2023,2024")
    search_parser.add_argument(
        "--preset",
        choices=("compact", "full", "organism"),
        default="compact",
    )
    search_parser.add_argument("--only", action="append", dest="only_candidates")
    search_parser.add_argument("--out", default=None)
    search_parser.add_argument("--resume", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "earthquake-v4":
        from hazardpulse.earthquake.v4_regional import main as run_earthquake_v4

        run_earthquake_v4(
            output_dir=args.output_dir,
            verbose=not args.quiet,
            honest=args.honest,
        )
        return 0

    if args.command == "hurricane":
        support = _load_hurricane_support()
        if support is None:
            parser.error("Hurricane utilities are not available in this checkout.")

        build_operational_ri_dataset = support["build_operational_ri_dataset"]
        summarize_operational_ri_cases = support["summarize_operational_ri_cases"]
        write_operational_ri_dataset = support["write_operational_ri_dataset"]
        default_feature_ablation_specs = support["default_feature_ablation_specs"]
        run_operational_ri_ablation_suite = support["run_operational_ri_ablation_suite"]
        select_feature_names_for_spec = support["select_feature_names_for_spec"]
        benchmark_operational_ri = support["benchmark_operational_ri"]
        load_operational_ri_cases = support["load_operational_ri_cases"]
        best_known_operational_ri_candidate = support["best_known_operational_ri_candidate"]
        default_operational_ri_tuning_candidates = support[
            "default_operational_ri_tuning_candidates"
        ]
        run_operational_ri_tuning_search = support["run_operational_ri_tuning_search"]

        if args.hurricane_command == "build-operational-ri-dataset":
            basins = _tuple_from_repeat(args.basins, ("AL",))
            aids = _tuple_from_repeat(args.aids, support["DEFAULT_AID_MODELS"])
            lead_times = _ints_from_csv(args.lead_times, support["DEFAULT_LEAD_TIMES"])

            cases = build_operational_ri_dataset(
                start_year=args.start_year,
                end_year=args.end_year,
                basin_prefixes=basins,
                aid_models=aids,
                lead_times=lead_times,
                limit_storms=args.limit_storms,
            )
            out_path = write_operational_ri_dataset(cases, Path(args.out))
            print(f"Wrote {len(cases)} operational RI cases to {out_path}")
            print(json.dumps(summarize_operational_ri_cases(cases), indent=2, sort_keys=True))
            return 0

        if args.hurricane_command == "benchmark-operational-ri":
            cases = load_operational_ri_cases(args.dataset)
            feature_names = None
            config = None
            candidate_name = None
            candidate_description = None
            if args.candidate != "raw_default":
                if args.candidate == "best_known":
                    candidate = best_known_operational_ri_candidate()
                else:
                    catalog = {
                        candidate.name: candidate
                        for preset in ("compact", "full", "organism")
                        for candidate in default_operational_ri_tuning_candidates(preset=preset)
                    }
                    candidate = catalog.get(args.candidate)
                    if candidate is None:
                        parser.error(f"Unknown benchmark candidate: {args.candidate}")
                feature_names = select_feature_names_for_spec(cases, candidate.feature_spec)
                config = candidate.config
                candidate_name = candidate.name
                candidate_description = candidate.description
            results = benchmark_operational_ri(
                cases,
                test_start_year=args.test_start_year,
                feature_names=feature_names,
                config=config,
            )
            if candidate_name is not None:
                results["benchmark_candidate"] = candidate_name
                results["benchmark_candidate_description"] = candidate_description
            print(json.dumps(results, indent=2, sort_keys=True))
            return 0

        if args.hurricane_command == "sweep-operational-ri":
            cases = load_operational_ri_cases(args.dataset)
            holdouts = tuple(int(part.strip()) for part in args.holdouts.split(",") if part.strip())
            specs = None
            if args.only_specs:
                available = default_feature_ablation_specs(preset=args.preset)
                wanted = {name.strip() for name in args.only_specs if name.strip()}
                specs = [spec for spec in available if spec.name in wanted]
                missing = sorted(wanted - {spec.name for spec in specs})
                if missing:
                    parser.error(
                        f"Unknown ablation spec(s) for preset {args.preset}: {', '.join(missing)}"
                    )
            results = run_operational_ri_ablation_suite(
                cases,
                holdouts=holdouts,
                specs=specs,
                preset=args.preset,
            )
            payload = json.dumps(results, indent=2, sort_keys=True)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(payload + "\n", encoding="utf-8")
                print(f"Wrote sweep results to {out_path}")
            print(payload)
            return 0

        if args.hurricane_command == "search-operational-ri":
            cases = load_operational_ri_cases(args.dataset)
            holdouts = tuple(int(part.strip()) for part in args.holdouts.split(",") if part.strip())
            candidates = None
            if args.only_candidates:
                available = default_operational_ri_tuning_candidates(preset=args.preset)
                wanted = {name.strip() for name in args.only_candidates if name.strip()}
                candidates = [candidate for candidate in available if candidate.name in wanted]
                missing = sorted(wanted - {candidate.name for candidate in candidates})
                if missing:
                    parser.error(
                        f"Unknown tuning candidate(s) for preset {args.preset}: "
                        f"{', '.join(missing)}"
                    )
            results = run_operational_ri_tuning_search(
                cases,
                holdouts=holdouts,
                candidates=candidates,
                preset=args.preset,
                checkpoint_path=args.out,
                resume=args.resume,
                log_fn=print,
            )
            payload = json.dumps(results, indent=2, sort_keys=True)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(payload + "\n", encoding="utf-8")
                print(f"Wrote search results to {out_path}")
            print(payload)
            return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
