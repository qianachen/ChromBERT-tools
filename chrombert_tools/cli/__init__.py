# -*- coding: utf-8 -*-
from __future__ import division, print_function
import click
import sys
from .. import __version__

# Monkey patch
click.core._verify_python3_env = lambda: None


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


@click.version_option(__version__, "-V", "--version")
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("-v", "--verbose", help="Verbose logging", is_flag=True, default=False)
@click.option(
    "-d", "--debug", help="Post mortem debugging", is_flag=True, default=False
)
def cli(verbose, debug):
    """
    Type -h or --help after any subcommand for more information.

    """
    if verbose:
        pass
        # logger.setLevel(logging.DEBUG)

    if debug:
        import traceback

        try:
            import ipdb as pdb
        except ImportError:
            import pdb

        def _excepthook(exc_type, value, tb):
            traceback.print_exception(exc_type, value, tb)
            print()
            pdb.pm()

        sys.excepthook = _excepthook


from . import (
    embed_region,
    embed_regulator,
    embed_cistrome,
    embed_gene,
    embed_cell_gene,
    embed_cell_cistrome,
    embed_cell_region,
    embed_cell_regulator,
    infer_trn,
    infer_cell_trn,
    # impute_cistrome,
    # find_driver_in_dual_region,
)

# Register all subcommands
cli.add_command(embed_region.embed_region)
cli.add_command(embed_regulator.embed_regulator)
cli.add_command(embed_cistrome.embed_cistrome)
cli.add_command(embed_gene.embed_gene)
cli.add_command(embed_cell_gene.embed_cell_gene)
cli.add_command(embed_cell_cistrome.embed_cell_cistrome)
cli.add_command(embed_cell_region.embed_cell_region)
cli.add_command(embed_cell_regulator.embed_cell_regulator)
cli.add_command(infer_trn.infer_trn)
cli.add_command(infer_cell_trn.infer_cell_trn)

