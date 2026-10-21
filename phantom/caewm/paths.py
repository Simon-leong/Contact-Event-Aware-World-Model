"""Default locations of the CAEWM data and artefacts, derived from configs/paths.yaml
(merged with the gitignored configs/paths.local.yaml):

    episodes      <data_root>/phantom-episodes/tasks
    artefacts     parent of <runs_root>, i.e. <runs_root>/.. holds cache/ and runs/
    tactile cache <artefacts>/cache/tactile
    latent cache  <artefacts>/cache/tactile_lat
    tokenizer     <runs_root>/tactile_ae/tactile_ae.pt
    eval windows  configs/eval/windows_val124x2_seed7.json (in the repo)

Every tool still accepts explicit flags; these are only the defaults."""
from __future__ import annotations

from pathlib import Path

from phantom.config.paths import REPO_ROOT, load_paths

_P = load_paths()
TASKS = _P.data_root / "phantom-episodes" / "tasks"
ARTEFACTS = _P.runs_root.parent
TAC_CACHE = ARTEFACTS / "cache" / "tactile"
LAT_CACHE = ARTEFACTS / "cache" / "tactile_lat"
TAC_AE = _P.runs_root / "tactile_ae" / "tactile_ae.pt"
EVAL_WINDOWS = REPO_ROOT / "configs" / "eval" / "windows_val124x2_seed7.json"


def episode_path(p) -> Path:
    """Window files store episodes relative to the task root."""
    p = Path(p)
    return p if p.is_absolute() else TASKS / p


def stored_or_default(stored, default) -> Path:
    """Checkpoints record the absolute paths of the machine they were trained on. Use the
    recorded path when it exists here, otherwise the default derived from configs/paths*.yaml."""
    if stored and Path(stored).exists():
        return Path(stored)
    return Path(default)
