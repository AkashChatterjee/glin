"""glin entry point: `glin serve` runs the MCP server over stdio."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from glin import __version__, server

HTTP_REMOVED = (
    "HTTP mode was removed in v0.1.4; v0.1.3 is the last release with it. "
    "glin runs over stdio only."
)


def _existing_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory {value!r} does not exist")
    return path


def _positive_mb(value: str) -> float:
    try:
        mb = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not mb > 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be greater than 0")
    return mb


def _parsers() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        prog="glin",
        description="glin: instant, exactly-explainable statistical classifiers "
        "for AI agents, over MCP.",
    )
    parser.add_argument("--version", action="version", version=f"glin-ml {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    serve = commands.add_parser(
        "serve",
        help="Run the glin MCP server over stdio, for local MCP clients.",
        description="Run the glin MCP server over stdio, for local MCP clients.",
    )
    serve.add_argument(
        "--allow-dir",
        dest="allow_dirs",
        action="append",
        type=_existing_dir,
        default=[],
        metavar="DIR",
        help="Directory train_model(csv_path=...) may read from; repeatable. "
        "Defaults to the home directory, plus the current directory unless it is "
        "/ or an ancestor of home.",
    )
    serve.add_argument(
        "--max-file-mb",
        type=_positive_mb,
        default=server.DEFAULT_MAX_FILE_MB,
        metavar="MB",
        help="Largest file train_model(csv_path=...) may read, in MB "
        f"(default: {server.DEFAULT_MAX_FILE_MB}).",
    )
    serve.add_argument(
        "--mode", choices=["stdio", "http", "sse"], default=None, help=argparse.SUPPRESS
    )
    return parser, serve


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser, serve = _parsers()
    args = parser.parse_args(argv)
    if args.mode in ("http", "sse"):
        serve.error(HTTP_REMOVED)
    if args.mode == "stdio":
        print(
            "glin: --mode is deprecated and ignored; remove it from the server args.",
            file=sys.stderr,
        )
    server.run_stdio(allowed_dirs=args.allow_dirs or None, max_file_mb=args.max_file_mb)


if __name__ == "__main__":
    main()
