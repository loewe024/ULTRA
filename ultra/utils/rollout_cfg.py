"""Added: reader of the teacher rollout config (ultra/data/cfg/teacher_rollout.yaml).

Shared by ultra/run_teacher_inference.py and scripts/render_rollout.py.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO / "ultra/data/cfg/teacher_rollout.yaml"
PATH_KEYS = ("checkpoint", "motion_dir", "output_dir", "source_dir", "quality_dir")
BOOL_KEYS = ("use_source_object", "videos", "skeleton")
INT_KEYS = ("quality_column", "seed")


def config_error(message):
    """Invalid settings: exit code 2, so that scripts/eval_teacher.sh stops before rendering."""
    print(message, file=sys.stderr)
    sys.exit(2)


def load_rollout_cfg(path=None):
    """Read and check a rollout config; paths relative to the repository root become absolute Paths.

    quality_dir may be null (teacher without the retarget quality observation); source_dir must exist when
    use_source_object or skeleton is set.
    """
    path = Path(path) if path else DEFAULT_CFG
    if not path.is_file():
        config_error(f"rollout config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in PATH_KEYS + BOOL_KEYS + INT_KEYS if k not in raw]
    if missing:
        config_error(f"{path}: missing {', '.join(missing)}")
    cfg = {"path": path.resolve()}
    for key in PATH_KEYS:
        value = raw[key]
        if value is None and key != "quality_dir":
            config_error(f"{path}: {key} must be set")
        cfg[key] = None if value is None else (REPO / value).resolve()
    for key in BOOL_KEYS:
        if not isinstance(raw[key], bool):
            config_error(f"{path}: {key} must be true or false")
        cfg[key] = raw[key]
    for key in INT_KEYS:
        if not isinstance(raw[key], int) or isinstance(raw[key], bool):
            config_error(f"{path}: {key} must be an integer")
        cfg[key] = raw[key]
    cfg = SimpleNamespace(**cfg)
    if not cfg.checkpoint.is_file():
        config_error(f"{path}: checkpoint not found: {cfg.checkpoint}")
    if not cfg.motion_dir.is_dir():
        config_error(f"{path}: motion_dir not found: {cfg.motion_dir}")
    if cfg.quality_dir is not None and not cfg.quality_dir.is_dir():
        config_error(f"{path}: quality_dir not found: {cfg.quality_dir}")
    if (cfg.use_source_object or cfg.skeleton) and not cfg.source_dir.is_dir():
        config_error(f"{path}: source_dir not found: {cfg.source_dir}")
    return cfg
