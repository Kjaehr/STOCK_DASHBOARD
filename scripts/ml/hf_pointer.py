from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone

try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except Exception:
    HAS_HF = False

# Central defaults (kept in sync with train_model_ensemble.py)
HF_HUB_REPO_ID_DEFAULT = os.getenv('HF_HUB_REPO_ID', 'Kjaehr/stock_dashboard_models')
HF_HUB_SUBDIR_DEFAULT = os.getenv('HF_HUB_SUBDIR', 'ensembles/')


def get_hf_pointer(
    source: str = 'local',
    latest_json_path: str | Path = 'ml_out/latest.json',
    repo_id: Optional[str] = None,
    subdir: Optional[str] = None,
    filename: str = 'latest.json',
) -> Tuple[str, str]:
    """
    Resolve (repo_id, path_in_repo) pointing to the latest uploaded model (.pkl) on Hugging Face Hub.

    - source='local': read ml_out/latest.json locally (written after CI upload)
    - source='hub': download latest.json directly from Hub (requires huggingface_hub)

    Returns: (repo_id, path_in_repo)
    """
    repo_id = repo_id or HF_HUB_REPO_ID_DEFAULT
    subdir = subdir or HF_HUB_SUBDIR_DEFAULT

    if source == 'local':
        p = Path(latest_json_path)
        if not p.is_file():
            raise FileNotFoundError(f"latest.json not found at {p}")
        data = json.loads(p.read_text())
        rid = data.get('repo_id', repo_id)
        path_in_repo = data.get('path_in_repo')
        if not path_in_repo:
            raise ValueError("latest.json missing 'path_in_repo'")
        return rid, path_in_repo

    elif source == 'hub':
        if not HAS_HF:
            raise RuntimeError("huggingface_hub not installed; cannot fetch from hub")
        # Try resolve ensembles/latest.json (or provided filename) from the model repo
        rel_path = f"{subdir}{filename}" if subdir else filename
        local_path = hf_hub_download(repo_id=repo_id, filename=rel_path, repo_type='model')
        with open(local_path, 'r') as f:
            data = json.load(f)
        rid = data.get('repo_id', repo_id)
        path_in_repo = data.get('path_in_repo')
        if not path_in_repo:
            raise ValueError("hub latest.json missing 'path_in_repo'")
        return rid, path_in_repo

    else:
        raise ValueError("source must be 'local' or 'hub'")


__all__ = [
    'get_hf_pointer',
    'HF_HUB_REPO_ID_DEFAULT',
    'HF_HUB_SUBDIR_DEFAULT',
]

