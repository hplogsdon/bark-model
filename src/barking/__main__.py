"""Console script for barking."""

import click


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx):
    if ctx.obj is None:
        ctx.obj = {}


@cli.command()
def run():
    pass


if __name__ == "__main__":
    cli()
