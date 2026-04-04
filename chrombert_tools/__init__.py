# -*- coding: utf-8 -*-
"""ChromBERT-tools: Command-line tools for ChromBERT"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nxviz')

__version__ = "1.0.0"

from .cli import make_dataset
from .cli import check_region_file
# from .api.embed_gene import embed_gene
from .api.embed_cistrome import embed_cistrome
from .api.embed_region import embed_region
from .api.embed_regulator import embed_regulator
# from .api.impute_cistrome import impute_cistrome
from .api.predict_tf_binding import predict_tf_binding
from .api.infer_regulator_network import infer_regulator_network
from .api.infer_ep import infer_ep

__all__ = [
    "__version__",
    # "embed_gene",
    "embed_cistrome",
    "embed_region",
    "embed_regulator",
    "infer_ep",
    # "impute_cistrome",
    "predict_tf_binding",
    "infer_regulator_network",
]