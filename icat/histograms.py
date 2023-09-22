"""Component to view the prediction distributions via histograms."""

from collections.abc import Callable

import ipyvuetify as v
import panel as pn

from icat.data import DataManager
from icat.histogram import Histogram
from icat.utils import _kill_param_auto_docstring

_kill_param_auto_docstring()


class Histograms(pn.viewable.Viewer):
    """Component class to show histogram distributions both of the
    current sample, as well as the across the entire active dataset.

    Args:
        width (int): Width to render the collection of histograms at.
    """

    def __init__(self, width: int = 700, **params):
        self.width = width

        # TODO: coupling: see note in log, but I think I'm okay with simply referencing
        # values in other components - it's triggering functionality that I'm concerned
        # about reducing.
        self.hist_local = Histogram(width=width - 25)
        self.hist_global = Histogram(width=width - 25)
        self.hist_global.layout.margin = (-35, 0, 0, 0)
        self.slider = v.RangeSlider(
            v_model=[0, 100],
            min=0,
            max=100,
            step=1,
            thumb_label=True,
            style_=f"padding: 0; padding-top: 35px; margin: 0; z-index: 10000; width: {width-25}px;",
        )
        self.slider.on_event("change", self._handle_ipv_range_changed)

        self.transparent_bg_css = v.Html(
            tag="style",
            children=[
                ".vuetify-styles .theme--dark.v-application {background: transparent}"
                ".vuetify-styles .theme--light.v-application {background: transparent}"
            ],
        )

        self._range_changed_callbacks: list[Callable] = []

        self._set_layout()
        super().__init__(**params)

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def _handle_ipv_range_changed(self, widget, event, data):
        data[0] /= 100
        data[1] /= 100
        self.fire_on_range_changed(data)

    # ============================================================
    # EVENT SPAWNERS
    # ============================================================

    def on_range_changed(self, callback: Callable):
        """Register a callback function for when the range slider is updated.

        Callbacks for this event should take an array of two elements [min, max]
        as a parameter.
        """
        self._range_changed_callbacks.append(callback)

    def fire_on_range_changed(self, data):
        for callback in self._range_changed_callbacks:
            callback(data)

    # ============================================================
    # INTERNAL FUNCTIONS
    # ============================================================

    def __panel__(self):
        return self.layout

    def _set_layout(self):
        self.layout = pn.Column(
            self.hist_local.layout,
            pn.Row(
                v.Container(
                    children=[self.slider, self.transparent_bg_css],
                    style_="padding: 0; margin: 0; overflow: hidden;",
                ),
                sizing_mode="stretch_width",
            ),
            self.hist_global.layout,
            width=self.width,
            height=227,
        )

    # ============================================================
    # PUBLIC FUNCTIONS
    # ============================================================

    def refresh_data(self, data: DataManager):
        """Update both local and global histograms for the currently
        active dataset in the datamanager.

        Args:
            data (DataManager): The data manager to pull the data from.
        """

        if data.active_data is None:
            return
        if data.prediction_col in data.active_data.columns:
            self.hist_local.set_data(
                data.active_data.loc[
                    data.sample_indices,
                    [data.prediction_col, data.label_col],
                ],
                data.prediction_col,
            )
            self.hist_global.set_data(
                data.active_data.loc[:, [data.prediction_col, data.label_col]],
                data.prediction_col,
            )
