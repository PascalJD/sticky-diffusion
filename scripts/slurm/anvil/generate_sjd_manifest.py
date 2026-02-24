#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _parse_float_csv(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def _parse_int_csv(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _label_float(x: float) -> str:
    s = f"{x:.6g}"
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def _anchor_flag(mode: str) -> str:
    return "true" if mode == "learned" else "false"


def _line_for(
    *,
    eta: float,
    p: float,
    temperature: float,
    anchor_mode: str,
    seed: int,
) -> str:
    label = (
        f"sjd_s1_eta{_label_float(eta)}"
        f"_p{_label_float(p)}"
        f"_temp{_label_float(temperature)}"
        f"_a{anchor_mode}"
        f"_s{seed}"
    )
    return " ".join(
        [
            f"experiment.forward.jump.eta={eta}",
            f"experiment.forward.hazard.p={p}",
            f"experiment.sampler.logit_temperature={temperature}",
            f"experiment.model.learnable_anchors={_anchor_flag(anchor_mode)}",
            f"experiment.training.seed={seed}",
            f"hydra.job.name={label}",
            f"wandb.run_name={label}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Stage-1 Slurm manifest lines for SJD eta x p tuning.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Manifest output path.")
    parser.add_argument(
        "--etas",
        default="0.9,0.85,0.8,0.75",
        help="CSV eta values.",
    )
    parser.add_argument(
        "--p-values",
        default="0.5,1,2,3",
        help="CSV polynomial p values.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Fixed sampler logit temperature for Stage-1.",
    )
    parser.add_argument(
        "--anchor-mode",
        default="fixed",
        choices=["fixed", "learned"],
        help="Anchor mode for Stage-1.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="CSV training seeds to include.",
    )
    args = parser.parse_args()

    etas = _parse_float_csv(args.etas)
    p_values = _parse_float_csv(args.p_values)
    seeds = _parse_int_csv(args.seeds)

    lines: list[str] = []
    for eta in etas:
        for p in p_values:
            for seed in seeds:
                lines.append(
                    _line_for(
                        eta=eta,
                        p=p,
                        temperature=float(args.temperature),
                        anchor_mode=str(args.anchor_mode),
                        seed=seed,
                    )
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {len(lines)} runs to {args.output}")


if __name__ == "__main__":
    main()
