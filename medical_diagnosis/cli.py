"""CLI for the multi-agent medical image diagnosis pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from medical_diagnosis.orchestrator import MedicalDiagnosisOrchestrator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-agent medical image diagnosis (GPT-4 vision + preprocessing + narratives/Q&A). "
        "Not for clinical use without validation and regulatory clearance."
    )
    parser.add_argument("image", type=Path, nargs="?", help="Path to JPEG/PNG/DICOM image")
    parser.add_argument(
        "--domain",
        choices=("auto", "radiology", "dermatology", "ophthalmology"),
        default="auto",
        help="Clinical pipeline to run (default: auto-routing agent)",
    )
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE enhancement during preprocessing")
    parser.add_argument("--health", action="store_true", help="Print model registry health JSON and exit")
    parser.add_argument(
        "--no-narratives",
        action="store_true",
        help="Skip GPT-4 lay summary, medical report, and contextual advice (vision agent only)",
    )
    parser.add_argument(
        "--patient-context",
        default=None,
        metavar="TEXT",
        help="Optional short non-PHI demo context for slightly tailored narrative recommendations",
    )
    parser.add_argument(
        "--ask",
        default=None,
        metavar="QUESTION",
        help="Question for healthcare-provider Q&A (after image run), grounded in that run's outputs",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        metavar="PATH",
        help="Prior JSON output from this tool; use with --ask for follow-up Q&A without re-running vision",
    )
    parser.add_argument(
        "--radiology-subspecialty",
        choices=("general", "breast", "neuro"),
        default=None,
        help="When domain is radiology or auto, force breast vs neuro vs general imaging specialist (default: infer)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    orch = MedicalDiagnosisOrchestrator(apply_clahe=args.clahe)
    if args.health:
        print(json.dumps(orch.registry.health_snapshot(), indent=2))
        return 0

    if args.bundle is not None:
        if not args.ask:
            print("Error: --ask is required when using --bundle", file=sys.stderr)
            return 1
        if not args.bundle.is_file():
            print(f"Error: bundle file not found: {args.bundle}", file=sys.stderr)
            return 1
        try:
            prior = json.loads(args.bundle.read_text(encoding="utf-8"))
            qa = orch.answer_question(prior, args.ask)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(json.dumps({"clinical_qa": qa}, indent=2))
        return 0

    if args.image is None:
        print("Error: image path is required unless using --health or --bundle", file=sys.stderr)
        return 1
    if not args.image.is_file():
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        return 1

    try:
        out = orch.run(
            args.image,
            mode=args.domain,
            with_narratives=not args.no_narratives,
            patient_context=args.patient_context,
            clinical_question=args.ask,
            radiology_subspecialty=args.radiology_subspecialty,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
