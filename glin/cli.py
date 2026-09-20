"""glin CLI: train, list, delete, serve."""
from __future__ import annotations

import shutil

import click
import pandas as pd

from glin import engine, server
from glin.engine import DEFAULT_MODELS_ROOT
from glin.validation import validate_dataset


@click.group()
def main() -> None:
    """glin: instant, exactly-explainable statistical classifiers for AI agents."""


def _print_training_progress(best_log_loss: float, elapsed_seconds: float) -> None:
    """engine.train_model's on_progress= callback: runs in the main process
    (from a watcher thread, not a training worker), so plain click.echo is
    safe here. Overwrites a single line in place rather than printing one
    line per bag per second -- "best validation log-loss across all bags
    so far," which is the number that actually matters for judging whether
    training is still making progress."""
    msg = f"  best log-loss so far: {best_log_loss:.4f}  ({elapsed_seconds:,.0f}s elapsed)"
    click.echo(f"\r{msg:<60}", nl=False)


@main.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", required=True, help="Name of the target column to predict.")
@click.option("--name", required=True, help="Unique name to save this model under.")
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress the live boosting-progress lines (stage messages still print).",
)
def train(csv_path: str, target: str, name: str, quiet: bool) -> None:
    """Train an EBM classifier on CSV_PATH and save it as --name."""
    click.echo(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    click.echo(f"  {len(df):,} rows, {len(df.columns)} columns")

    click.echo("Validating dataset...")
    validation = validate_dataset(df, target)
    if validation.warnings:
        click.echo("Warnings:")
        for issue in validation.warnings:
            click.echo(f"  - {issue.message}")
        click.echo()
    if not validation.is_valid:
        click.echo("Errors:")
        for issue in validation.errors:
            click.echo(f"  - {issue.message}")
        raise SystemExit(1)

    click.echo(
        f"Training EBM classifier on {len(df):,} rows "
        "(large datasets can take a few minutes)..."
    )
    bundle = engine.train_model(
        df,
        target,
        model_name=name,
        on_progress=None if quiet else _print_training_progress,
    )
    if not quiet:
        click.echo()  # close out the in-place progress line

    click.echo("Saving model...")
    model_dir = engine.save_bundle(bundle, DEFAULT_MODELS_ROOT)

    preprocessor = bundle["preprocessor"]
    dropped = ", ".join(preprocessor.dropped_columns_) or "none"
    click.echo(f"Trained '{name}' -> {model_dir}")
    click.echo(f"  Classes:          {', '.join(str(c) for c in bundle['target_classes'])}")
    click.echo(
        f"  Features kept:    {len(preprocessor.output_columns_)} "
        f"({len(preprocessor.numeric_columns_)} numeric, "
        f"{len(preprocessor.categorical_columns_)} categorical)"
    )
    click.echo(f"  Features dropped: {dropped}")


@main.command(name="list")
def list_cmd() -> None:
    """List all trained models."""
    if not DEFAULT_MODELS_ROOT.exists():
        click.echo("No models trained yet.")
        return

    rows = []
    for model_dir in sorted(DEFAULT_MODELS_ROOT.iterdir()):
        metadata_path = model_dir / engine.METADATA_FILENAME
        if not metadata_path.exists():
            continue
        meta = engine.load_metadata(model_dir)
        n_features = len(meta["features"]["numeric"]) + len(meta["features"]["categorical"])
        rows.append(
            (
                meta["model_name"],
                meta["target_column"],
                ", ".join(str(c) for c in meta["target_classes"]),
                str(n_features),
                meta["created_at"],
            )
        )

    if not rows:
        click.echo("No models trained yet.")
        return

    headers = ("Model Name", "Target Column", "Classes", "Features Kept", "Created Date")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(values: tuple[str, ...]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, widths))

    click.echo(fmt_row(headers))
    click.echo(fmt_row(tuple("-" * w for w in widths)))
    for row in rows:
        click.echo(fmt_row(row))


@main.command()
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def delete(name: str, yes: bool) -> None:
    """Delete a trained model."""
    model_dir = DEFAULT_MODELS_ROOT / name
    if not model_dir.exists():
        click.echo(f"Error: model '{name}' not found in {DEFAULT_MODELS_ROOT}")
        raise SystemExit(1)

    if not yes:
        click.confirm(f"Delete model '{name}' at {model_dir}?", abort=True)

    shutil.rmtree(model_dir)
    click.echo(f"Deleted model '{name}'")


@main.command()
@click.option("--mode", type=click.Choice(["stdio", "http"]), default="stdio")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def serve(mode: str, host: str, port: int) -> None:
    """Run the glin MCP server (stdio for local clients, http for remote agents)."""
    if mode == "stdio":
        server.run_stdio()
    else:
        server.run_http(host=host, port=port)


if __name__ == "__main__":
    main()
