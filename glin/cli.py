"""glin CLI: train, list, serve."""
from __future__ import annotations

import click
import pandas as pd

from glin import engine, server
from glin.engine import DEFAULT_MODELS_ROOT
from glin.validation import validate_dataset


@click.group()
def main() -> None:
    """glin: instant, exactly-explainable statistical classifiers for AI agents."""


@main.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", required=True, help="Name of the target column to predict.")
@click.option("--name", required=True, help="Unique name to save this model under.")
def train(csv_path: str, target: str, name: str) -> None:
    """Train an EBM classifier on CSV_PATH and save it as --name."""
    df = pd.read_csv(csv_path)

    validation = validate_dataset(df, target)
    for issue in validation.warnings:
        click.echo(f"Warning: {issue.message}")
    if not validation.is_valid:
        for issue in validation.errors:
            click.echo(f"Error: {issue.message}")
        raise SystemExit(1)

    bundle = engine.train_model(df, target, model_name=name)
    model_dir = engine.save_bundle(bundle, DEFAULT_MODELS_ROOT)

    preprocessor = bundle["preprocessor"]
    click.echo(f"Trained model '{name}' -> {model_dir}")
    click.echo(f"  Classes: {bundle['target_classes']}")
    click.echo(
        f"  Features kept: {len(preprocessor.output_columns_)} "
        f"({len(preprocessor.numeric_columns_)} numeric, "
        f"{len(preprocessor.categorical_columns_)} categorical)"
    )
    click.echo(f"  Features dropped: {preprocessor.dropped_columns_}")


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
