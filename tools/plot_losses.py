"""Create loss plots from logs written by the vendored reference trainer."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LOSS_NAMES = ("total", "mel", "mel postnet", "pitch", "energy", "duration")
VALUE = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
TRAIN_RE = re.compile(
    rf"Step\s+(\d+)/\d+,\s+Total Loss:\s*{VALUE},\s*Mel Loss:\s*{VALUE},\s*"
    rf"Mel PostNet Loss:\s*{VALUE},\s*Pitch Loss:\s*{VALUE},\s*Energy Loss:\s*{VALUE},\s*Duration Loss:\s*{VALUE}"
)
VALIDATION_RE = re.compile(
    rf"Validation Step\s+(\d+),\s+Total Loss:\s*{VALUE},\s*Mel Loss:\s*{VALUE},\s*"
    rf"Mel PostNet Loss:\s*{VALUE},\s*Pitch Loss:\s*{VALUE},\s*Energy Loss:\s*{VALUE},\s*Duration Loss:\s*{VALUE}"
)


def _parse(path: Path, pattern: re.Pattern[str]) -> tuple[list[int], list[list[float]]]:
    if not path.is_file():
        return [], [[] for _ in LOSS_NAMES]
    steps: list[int] = []
    columns: list[list[float]] = [[] for _ in LOSS_NAMES]
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.search(line)
            if not match:
                continue
            steps.append(int(match.group(1)))
            for column, value in zip(columns, match.groups()[1:]):
                column.append(float(value))
    return steps, columns


def _plot(path: Path, steps: list[int], columns: list[list[float]], title: str) -> None:
    if not steps:
        raise ValueError(f"No parseable loss points found for {title}; log: {path}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    colors = ("#202A44", "#007C91", "#4C956C", "#DB7C26", "#A23B72", "#6C5B7B")
    for label, values, color in zip(LOSS_NAMES, columns, colors):
        ax.plot(steps, values, label=label.title(), linewidth=1.5, color=color)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--validation-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_steps, train_columns = _parse(args.training_log, TRAIN_RE)
    validation_steps, validation_columns = _parse(args.validation_log, VALIDATION_RE)
    _plot(
        args.output_dir / "training_loss.png",
        train_steps,
        train_columns,
        f"{args.language} FastSpeech 2 Training Loss",
    )
    _plot(
        args.output_dir / "validation_loss.png",
        validation_steps,
        validation_columns,
        f"{args.language} FastSpeech 2 Validation Loss",
    )
    print(
        f"Saved actual log-based loss plots: {len(train_steps)} training points, "
        f"{len(validation_steps)} validation points."
    )


if __name__ == "__main__":
    main()
