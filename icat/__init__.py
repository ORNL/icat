# flake8: noqa

# make all submodules directly accessible from a single curifactory import
from icat import (
    anchorlist,
    anchors,
    data,
    histogram,
    histograms,
    instance,
    model,
    table,
    view,
)
from icat.anchors import DictionaryAnchor, SimilarityFunctionAnchor, TFIDFAnchor

# make the important things directly accessible off top level module
from icat.model import Model

__version__ = "0.3.0"
