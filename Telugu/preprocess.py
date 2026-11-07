"""Audit, phonemize, align, and extract Telugu FastSpeech 2 training features."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [
    str(ROOT / "tools" / "prepare_pipeline.py"),
    "--config",
    str(Path(__file__).with_name("config.yaml")),
    *sys.argv[1:],
]
runpy.run_path(str(ROOT / "tools" / "prepare_pipeline.py"), run_name="__main__")
