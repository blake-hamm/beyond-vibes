"""HuggingFace client for listing and downloading files."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from beyond_vibes.model_config import ESSENTIAL_MODEL_CONFIGS


class HFClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self, token: str | None = None) -> None:
        """Initialize the HF client."""
        self._api = HfApi(token=token)
        self._token = token

    def list_files(self, repo_id: str, revision: str = "main") -> list[str]:
        """List all files in a repository."""
        return self._api.list_repo_files(repo_id=repo_id, revision=revision)

    def filter_files(self, files: list[str], quant_tags: list[str]) -> list[str]:
        """Filter files by quant tags or essential config files."""
        result = []
        for f in files:
            filename = Path(f).name
            if filename in ESSENTIAL_MODEL_CONFIGS:
                result.append(f)
            elif any(tag.lower() in filename.lower() for tag in quant_tags):
                result.append(f)
        return result

    def download_file(self, repo_id: str, revision: str, filename: str) -> Path:
        """Download a single file from the repository."""
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                token=self._token,
            )
        )
