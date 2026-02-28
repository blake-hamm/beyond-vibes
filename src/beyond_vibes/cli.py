"""CLI for beyond vibes."""

import json
import logging
from pathlib import Path

import typer

from beyond_vibes.model_config import ModelConfig, load_models_config
from beyond_vibes.model_downloader import HFClient, S3Client
from beyond_vibes.settings import settings
from beyond_vibes.simulations import SimulationLogger
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.opencode import OpenCodeClient
from beyond_vibes.simulations.orchestration import run_simulation
from beyond_vibes.simulations.prompts.loader import build_prompt, load_prompt
from beyond_vibes.simulations.sandbox import SandboxManager

logger = logging.getLogger(__name__)

app = typer.Typer()
DEFAULT_CONFIG = "models.yaml"


@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """CLI entry point with optional debug flag."""
    if debug:
        logging.getLogger("beyond_vibes").setLevel(logging.DEBUG)


@app.command()
def download(
    config_path: Path | None = Path(DEFAULT_CONFIG),
    dry_run: bool = False,
) -> None:
    """Download models from HuggingFace to S3."""
    config = load_models_config(config_path)

    s3_client = S3Client()
    hf_client = HFClient(token=settings.hf_token)

    for model in config.models:
        if model.repo_id is None:
            logger.info(
                f"Skipping {model.name} - provider '{model.provider}' "
                "does not require download"
            )
            continue

        logger.info(f"Processing {model.name}...")

        try:
            files = hf_client.list_files(model.repo_id, model.revision)
        except Exception as e:
            logger.error(f"Failed to list files for {model.name}: {e}")
            raise typer.Exit(code=1) from e

        filtered = hf_client.filter_files(files, model.quant_tags)

        logger.info(f"  Found {len(filtered)} files to download")

        for f in filtered:
            s3_key = f"{model.name}/{model.repo_id}/{f}"

            if dry_run:
                logger.info(
                    f"  [DRY RUN] Would upload to s3://{settings.s3_bucket}/{s3_key}"
                )
            else:
                try:
                    local_path = hf_client.download_file(
                        model.repo_id, model.revision, f
                    )
                except Exception as e:
                    logger.error(f"Failed to download {f}: {e}")
                    raise typer.Exit(code=1) from e

                try:
                    s3_client.upload_file(local_path, s3_key)
                except Exception as e:
                    logger.error(f"Failed to upload {f}: {e}")
                    raise typer.Exit(code=1) from e

                logger.info(
                    f"  Successfully downloaded {f} and uploaded to s3 {s3_key}"
                )

    logger.info("Done!")


@app.command()
def simulate(  # noqa: PLR0913
    task: str = typer.Option(..., "--task", help="Task name (without .yaml)"),
    model: str | None = typer.Option(
        None, "--model", help="Model name from models.yaml"
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Filter by provider (e.g., local, openai)"
    ),
    config_path: Path | None = Path(DEFAULT_CONFIG),
    prompt_vars: str = typer.Option(
        "{}", "--prompt-vars", help="JSON dict of variables"
    ),
    quant: str | None = typer.Option(
        None,
        "--quant",
        help="Quantization tag (e.g., Q6_K_XL). Uses first if not specified.",
    ),
) -> None:
    """Run a simulation by cloning a repo and executing a prompt via OpenCode."""
    sim_config = _load_task_config(task, prompt_vars)
    prompt = build_prompt(sim_config)

    # Validate that at least one of model or provider is specified
    if model is None and provider is None:
        logger.error("Must specify either --model or --provider")
        raise typer.Exit(code=1)

    # Get models to run
    models_to_run = _get_models_to_run(model, provider, config_path)

    # Run simulation for each model
    error_occurred = False
    for model_config in models_to_run:
        logger.info(f"Running simulation with model: {model_config.name}")

        quant_tag = quant or (
            model_config.quant_tags[0] if model_config.quant_tags else None
        )

        sandbox = SandboxManager()

        with OpenCodeClient() as opencode_client:
            sim_logger = SimulationLogger(quant_tag=quant_tag)

            model_error = run_simulation(
                sim_config, model_config, sandbox, opencode_client, sim_logger, prompt
            )
            error_occurred = error_occurred or model_error

        sandbox.cleanup()
        logger.info("Sandbox cleaned up")

    if error_occurred:
        raise typer.Exit(code=1)


def _load_model_config(model: str, config_path: Path | None) -> ModelConfig:
    """Load and validate model config from models.yaml."""
    config = load_models_config(config_path)

    for m in config.models:
        if m.name == model:
            return m

    logger.error(f"Model '{model}' not found in {config_path}")
    raise typer.Exit(code=1)


def _get_models_to_run(
    model: str | None, provider: str | None, config_path: Path | None
) -> list[ModelConfig]:
    """Get list of models to run based on filters."""
    config = load_models_config(config_path)
    models = []

    for m in config.models:
        # Filter by model name if specified
        if model is not None and m.name != model:
            continue
        # Filter by provider if specified
        if provider is not None and m.provider != provider:
            continue
        models.append(m)

    if not models:
        filters = []
        if model:
            filters.append(f"model='{model}'")
        if provider:
            filters.append(f"provider='{provider}'")
        logger.error(f"No models found matching filters: {', '.join(filters)}")
        raise typer.Exit(code=1)

    return models


def _load_task_config(task: str, prompt_vars: str) -> SimulationConfig:
    """Load and parse task prompt configuration."""
    variables = json.loads(prompt_vars)

    prompts_dir = Path(__file__).parent / "simulations" / "prompts" / "tasks"
    prompt_path = prompts_dir / f"{task}.yaml"

    try:
        return load_prompt(prompt_path, variables)
    except FileNotFoundError:
        logger.error(f"Task not found: {prompt_path}")
        raise typer.Exit(code=1) from None
    except Exception as e:
        logger.error(f"Failed to load prompt: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
