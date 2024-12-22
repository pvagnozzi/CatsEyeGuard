"""Command Line Interface."""

import click


@click.group()
def cli() -> None:
    """CLI for CatsEyeGuard."""


@cli.command()
def run() -> None:
    """Run command."""
