"""CLI for beyond vibes."""

import json
import logging
from pathlib import Path

import typer
import yaml

from beyond_vibes.model_downloader import Config, HFClient, S3Client
from beyond_vibes.opencode_client import OpenCodeClient
from beyond_vibes.settings import settings
from beyond_vibes.simulations import SimulationLogger
from beyond_vibes.simulations.prompts.loader import load_prompt
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
    config_data = yaml.safe_load(config_path.read_text())
    config = Config(**config_data)

    s3_client = S3Client()
    hf_client = HFClient(token=settings.hf_token)

    for model in config.models:
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
def simulate(
    task: str = typer.Option(..., "--task", help="Task name (without .yaml)"),
    prompt_vars: str = typer.Option(
        "{}", "--prompt-vars", help="JSON dict of variables"
    ),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Skip sandbox cleanup"),
) -> None:
    """Run a simulation by cloning a repo and executing a prompt via OpenCode."""
    prompts_dir = Path(__file__).parent / "simulations" / "prompts" / "tasks"
    prompt_path = prompts_dir / f"{task}.yaml"

    try:
        variables = json.loads(prompt_vars)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in --prompt-vars: {e}")
        raise typer.Exit(code=1) from e

    try:
        sim_config = load_prompt(prompt_path, variables)
    except FileNotFoundError as e:
        logger.error(f"Task not found: {prompt_path}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error(f"Failed to load prompt: {e}")
        raise typer.Exit(code=1) from e

    sandbox = SandboxManager()
    opencode_client = OpenCodeClient()
    sim_logger = SimulationLogger()

    error_occurred = False
    try:
        with sim_logger.log_simulation(sim_config) as logger_ctx:
            with sandbox.sandbox(
                url=sim_config.repository.url,
                branch=sim_config.repository.branch,
            ) as working_dir:
                if working_dir is None:
                    raise RuntimeError("Failed to create sandbox")

                logger.info(f"Running simulation '{sim_config.name}' in {working_dir}")

                session_id = opencode_client.create_session(working_dir)

                response = opencode_client.run_prompt(session_id, sim_config.prompt)

                logger_ctx.log_turn(
                    turn_index=0,
                    response=str(response),
                )

                logger.info("Simulation completed successfully")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        error_occurred = True
        if sim_logger.session:
            sim_logger.log_error(str(e))

    finally:
        if no_cleanup:
            logger.info(f"Sandbox preserved at: {sandbox.path}")
        else:
            sandbox.cleanup()
            logger.info("Sandbox cleaned up")

    if error_occurred:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
