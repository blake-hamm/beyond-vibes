"""CLI for beyond vibes."""

import logging
from pathlib import Path

import typer

from beyond_vibes.evaluations.extractor import query_simulation_runs
from beyond_vibes.evaluations.runner import EvaluationRunner
from beyond_vibes.model_config import (
    get_model_by_name,
    load_models_config,
)
from beyond_vibes.model_downloader import HFClient, S3Client
from beyond_vibes.settings import settings
from beyond_vibes.simulations import SimulationLogger
from beyond_vibes.simulations.orchestration import run_simulation
from beyond_vibes.simulations.pi_dev import PiDevClient
from beyond_vibes.simulations.prompts.loader import build_prompt, load_task_config
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
    model: str = typer.Option(..., "--model", help="Model name from models.yaml"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider filter when multiple models have the same name",
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
    try:
        sim_config = load_task_config(task, prompt_vars)
    except FileNotFoundError:
        logger.error(f"Task not found: {task}")
        raise typer.Exit(code=1) from None
    except Exception as e:
        logger.error(f"Failed to load prompt: {e}")
        raise typer.Exit(code=1) from None

    prompt = build_prompt(sim_config)

    # Get the single model to run (optionally filtered by provider)
    try:
        model_config = get_model_by_name(model, provider, config_path)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1) from None

    logger.info(f"Running simulation with model: {model_config.name}")

    quant_tag = quant or (
        model_config.quant_tags[0] if model_config.quant_tags else None
    )

    sandbox = SandboxManager()

    pi_client = PiDevClient(
        provider=model_config.provider,
        model=model_config.get_model_id(),
        timeout=settings.simulation_timeout,
    )
    sim_logger = SimulationLogger(quant_tag=quant_tag)

    try:
        run_simulation(sim_config, model_config, sandbox, pi_client, sim_logger, prompt)
    except Exception as e:
        sandbox.cleanup()
        logger.info("Sandbox cleaned up")
        logger.error("Simulation failed: %s", e)
        raise typer.Exit(code=1) from e

    sandbox.cleanup()
    logger.info("Sandbox cleaned up")


@app.command()
def evaluate(  # noqa: PLR0912,PLR0913
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Specific run to evaluate (if omitted, evaluates all matching filters)",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Filter by task name",
    ),
    archetype: str | None = typer.Option(
        None,
        "--archetype",
        "-a",
        help="Filter by archetype",
    ),
    experiment: str = typer.Option(
        "beyond-vibes",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        "-m",
        help="Override judge model (e.g., 'gpt-4o', 'local-model')",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Show what would be evaluated without running",
    ),
) -> None:
    """Evaluate simulation runs using configured judges.

    Evaluates runs from MLflow using judges defined in task configuration.
    Results are logged back to the original runs.

    Examples
    --------
        # Evaluate specific run
        beyond-vibes evaluate --run-id abc123

        # Evaluate all runs for a task
        beyond-vibes evaluate --task poetry_to_uv

        # Evaluate with different model
        beyond-vibes evaluate --run-id abc123 --judge-model gpt-4o

        # Dry run to see what would be evaluated
        beyond-vibes evaluate --task poetry_to_uv --dry-run

    """
    # Initialize runner
    runner = EvaluationRunner(judge_model=judge_model)

    if run_id:
        # Single run evaluation
        logger.info(f"Evaluating single run: {run_id}")

        if dry_run:
            typer.echo(f"Would evaluate run: {run_id}")
            return

        try:
            results = runner.evaluate_run(run_id)

            # Display results
            typer.echo(f"✓ Evaluated run {run_id}")
            for judge_name, result in results.items():
                if "error" in result:
                    typer.echo(f"  ✗ {judge_name}: ERROR - {result['error']}")
                else:
                    score = result.get("score", 0.0)
                    typer.echo(f"  ✓ {judge_name}: {score:.2f}")

        except Exception as e:
            logger.error(f"Failed to evaluate run {run_id}: {e}")
            raise typer.Exit(1) from e

    else:
        # Batch evaluation
        logger.info(f"Querying runs from experiment: {experiment}")

        try:
            runs = query_simulation_runs(
                experiment=experiment,
                task_name=task,
                archetype=archetype,
            )
        except Exception as e:
            logger.error(f"Failed to query runs: {e}")
            raise typer.Exit(1) from e

        if not runs:
            typer.echo("No runs found matching criteria")
            raise typer.Exit(0)

        typer.echo(f"Found {len(runs)} runs to evaluate")

        if dry_run:
            for run in runs:
                typer.echo(f"  Would evaluate: {run.info.run_id}")
            return

        # Evaluate all runs
        success_count = 0
        error_count = 0

        with typer.progressbar(runs, label="Evaluating") as progress:
            for run in progress:
                try:
                    results = runner.evaluate_run(run.info.run_id)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to evaluate run {run.info.run_id}: {e}")
                    error_count += 1

        # Summary
        typer.echo("\nEvaluation complete:")
        typer.echo(f"  ✓ Successful: {success_count}")
        if error_count > 0:
            typer.echo(f"  ✗ Failed: {error_count}")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
