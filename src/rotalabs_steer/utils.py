"""Utility functions for rotalabs-steer."""

from __future__ import annotations

import shutil
from pathlib import Path

# Default HuggingFace repository for pre-extracted vectors
DEFAULT_VECTORS_REPO = "rotalabs/steering-vectors"


def download_vectors(
    behavior: str,
    model_name: str,
    output_dir: Path | None = None,
    repo_id: str = DEFAULT_VECTORS_REPO,
    revision: str = "main",
    force: bool = False,
) -> Path:
    """
    Download pre-extracted steering vectors from HuggingFace Hub.

    Args:
        behavior: Behavior type (e.g., 'refusal', 'uncertainty', 'tool_restraint')
        model_name: Model the vectors were extracted from (e.g., 'qwen3-8b')
        output_dir: Local directory to save vectors (default: ~/.cache/rotalabs_steer/vectors)
        repo_id: HuggingFace repository ID
        revision: Git revision (branch/tag/commit)
        force: Re-download even if vectors exist locally

    Returns:
        Path to downloaded vector directory

    Example:
        ```python
        from rotalabs_steer.utils import download_vectors
        from rotalabs_steer import SteeringVectorSet

        # Download refusal vectors for Qwen3-8B
        vector_path = download_vectors("refusal", "qwen3-8b")

        # Load the vectors
        vectors = SteeringVectorSet.load(vector_path)
        ```
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError(
            "huggingface_hub package required for downloading vectors. "
            "Install with: pip install huggingface_hub"
        ) from err

    # default output directory
    if output_dir is None:
        cache_dir = Path.home() / ".cache" / "rotalabs_steer" / "vectors"
    else:
        cache_dir = Path(output_dir)

    # create output path
    local_path = cache_dir / model_name / behavior
    local_path.mkdir(parents=True, exist_ok=True)

    # check if already downloaded
    metadata_file = local_path / "metadata.json"
    if metadata_file.exists() and not force:
        return local_path

    # download from HuggingFace
    subfolder = f"{model_name}/{behavior}"

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=cache_dir,
            allow_patterns=[f"{subfolder}/*"],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download vectors for {behavior}/{model_name} from {repo_id}: {e}"
        ) from e

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Downloaded files but metadata.json not found at {local_path}. "
            f"The requested vectors may not exist in the repository."
        )

    return local_path


def list_available_vectors(
    repo_id: str = DEFAULT_VECTORS_REPO,
) -> list[dict]:
    """
    List available pre-extracted vectors in the HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        List of available vector configurations

    Example:
        ```python
        from rotalabs_steer.utils import list_available_vectors

        vectors = list_available_vectors()
        for v in vectors:
            print(f"{v['model']}/{v['behavior']}: layers {v['layers']}")
        ```
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as err:
        raise ImportError(
            "huggingface_hub package required. Install with: pip install huggingface_hub"
        ) from err

    api = HfApi()

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        raise RuntimeError(f"Failed to list files in {repo_id}: {e}") from e

    # parse file structure to find available vectors
    vectors = []
    metadata_files = [f for f in files if f.endswith("metadata.json")]

    for meta_file in metadata_files:
        parts = meta_file.split("/")
        if len(parts) >= 3:
            model = parts[-3]
            behavior = parts[-2]
            vectors.append({
                "model": model,
                "behavior": behavior,
                "path": f"{model}/{behavior}",
            })

    return vectors


def get_cache_dir() -> Path:
    """Get the default cache directory for rotalabs-steer."""
    return Path.home() / ".cache" / "rotalabs_steer"


def clear_cache(vectors_only: bool = True) -> None:
    """
    Clear the rotalabs-steer cache.

    Args:
        vectors_only: If True, only clear downloaded vectors. If False, clear entire cache.
    """
    cache_dir = get_cache_dir()

    if vectors_only:
        vectors_dir = cache_dir / "vectors"
        if vectors_dir.exists():
            shutil.rmtree(vectors_dir)
    else:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
