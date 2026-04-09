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
        
# from .utils import make_dataset
from .utils import check_region_file

from . import (
    embed_region,
    embed_regulator,
    predict_region_function_classification,
    predict_region_activity_regression,
    predict_gene_expression,
    interpret_region_region_interactions,
    interpret_regulator_regulator_interactions,
    interpret_regulator_effects_between_regions_groups,
    find_dirver_in_transition,
    find_cell_key_regulator,
    find_regulator_context_cofactors,
    impute_cistrome
)

# Register all subcommands
cli.add_command(embed_region.embed_region)
cli.add_command(embed_regulator.embed_regulator)

cli.add_command(predict_region_function_classification.predict_region_function_classification)
cli.add_command(predict_region_activity_regression.predict_region_activity_regression)
cli.add_command(predict_gene_expression.predict_gene_expression)

cli.add_command(interpret_region_region_interactions.interpret_region_region_interactions)
cli.add_command(interpret_regulator_regulator_interactions.interpret_regulator_regulator_interactions)
cli.add_command(interpret_regulator_effects_between_regions_groups.interpret_regulator_effects_between_regions_groups)

cli.add_command(find_dirver_in_transition.find_dirver_in_transition)
cli.add_command(find_cell_key_regulator.find_cell_key_regulator)
cli.add_command(find_regulator_context_cofactors.find_regulator_context_cofactors)
cli.add_command(impute_cistrome.impute_cistrome)
