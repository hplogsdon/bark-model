import click


@click.group()
@click.pass_context
def cli(ctx):
    if ctx.obj is None:
        ctx.obj = dict()


@cli.command()
def load(ctx):
    print(ctx.obj)