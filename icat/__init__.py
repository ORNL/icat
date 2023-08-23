# flake8: noqa
import panel as pn

# make all submodules directly accessible from a single curifactory import
from icat import (
    anchorlist,
    anchors,
    data,
    histogram,
    histograms,
    item,
    model,
    table,
    view,
)
from icat.anchors import Anchor, DictionaryAnchor, SimilarityFunctionAnchor, TFIDFAnchor

# make the important things directly accessible off top level module
from icat.model import Model

__version__ = "0.6.0"


def initialize(offline: bool = False):
    """Set up panel and ICAT-specific stylesheets.

    Call this function before running any ICAT model ``.view`` cells.
    If you want to handle initialization yourself, panel needs the "vega" extension:

    .. code-block:: python

        import panel as pn
        pn.extension('vega')

    Note that there's a weird conflict between ipyvuetify and panel where if you don't
    run this initialize function, you may need to run your first ``model.view`` cell twice
    for the stylesheets to correctly apply to some of the ipyvuetify datatables.

    Args:
        offline (bool): If set to true, will configure panel to draw js/css resources from
            local packages rather than hitting a CDN for them.
    """
    pn.extension("vega", inline=offline)

    # pre render the anchorlist template once (invisibly) so that the css gets loaded
    # (there's a weird conflict between the ghost dom that panel uses and how ipyvuetify
    # components seem to get their ghost dom styling.)
    # Yes this is a ridiculous hack. No, I no longer care, I've spent way too much time
    # trying to make it so the user doesn't have to execute their first model.view cell twice.
    from icat.anchorlist import AnchorListTemplate
    from icat.table import TableContentsTemplate

    return pn.Row(
        AnchorListTemplate(), TableContentsTemplate(), styles={"display": "none"}
    )
