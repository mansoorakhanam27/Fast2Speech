"""Synthesize one Telugu sentence and save its waveform and variance plots."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [
    str(ROOT / "tools" / "run_reference_inference.py"),
    "--config",
    str(Path(__file__).with_name("config.yaml")),
    *sys.argv[1:],
]
runpy.run_path(str(ROOT / "tools" / "run_reference_inference.py"), run_name="__main__")
