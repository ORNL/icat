"""Panel/holoviz code for a histogram distribution of predicted values."""

# we use altair because there appears to be a bug when directly using vega
# in panel - you can't update the graph after it's been created. (Despite
# what their documentation shows.)

import altair as alt
import numpy as np
import pandas as pd
import panel as pn

from icat.utils import _kill_param_auto_docstring

_kill_param_auto_docstring()


class Histogram(pn.viewable.Viewer):
    """Histogram to show the distribution of prediction outputs from the model
    on some set of the data. Anything above .5 is appropriate colored to orange for
    interesting.

    This is done via a Vega pane.

    Args:
        width (int): How wide to render the histogram.
    """

    def __init__(self, width: int = 400, **params):
        self.data = pd.DataFrame()
        self.width = width
        self.layout = pn.panel(
            self.get_vega_graph(), theme="dark", margin=(0, 0, -37, 0)
        )
        super().__init__(**params)

    def _compute_bins(self, df: pd.DataFrame, prediction_col: str):
        """We compute histogram values ourselves - even though altair/vega _can_
        (and normally would) do this for us, altair artificially limits the number
        of points it can handle to 5000, primarily because otherwise your notebook
        can accidentally become huge and impact performance (presumably from the JS
        side?)"""
        bins = np.linspace(0.0, 1.0, 51)
        labels = bins[:-1]
        hist_df = pd.DataFrame(
            {
                "count": pd.cut(
                    df[prediction_col], bins=bins, labels=labels
                ).value_counts()
            }
        )
        hist_df["bin"] = hist_df.index

        # make sure to include the 'width' of each bin, which is just the starting value of the next bin
        hist_df["bin_end"] = hist_df["bin"].astype(float) + 0.02
        hist_df["bin_end"] = hist_df.bin_end.apply(round, ndigits=2)
        hist_df["bin"] = hist_df.bin.apply(round, ndigits=2)
        hist_df.set_index(np.array(list(hist_df.bin)), inplace=True)
        self.data = hist_df
        # self.data = hist_df.to_dict(orient='records')

    def set_data(self, df: pd.DataFrame, prediction_col: str):
        """Re-generate the histogram given the passed data (which should
        already include the prediction values.)

        Args:
            df (pd.DataFrame): The dataset with a column of prediction outputs in it.
            prediction_col (str): The name of the column with prediction outputs.
        """
        self._compute_bins(df, prediction_col)
        self.layout.object = self.get_vega_graph()

    def get_vega_graph(self) -> alt.Chart:
        """Use altair to create the bar chart for the current data."""
        data = self.data
        chart = (
            alt.Chart(data)
            .mark_bar(align="left")
            .encode(
                x=alt.X(
                    "bin:Q",
                    title="",
                    bin="binned",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(
                        values=[float(i) / 10 for i in range(11)],
                        labels=True,
                        labelColor="white",
                        domainColor="white",
                        gridColor="#444",
                    ),
                ),
                x2="bin_end:Q",
                y=alt.Y(
                    "count:Q",
                    title="",
                    axis=alt.Axis(labels=False, domainColor="white", gridColor="#444"),
                ),
                color=alt.condition(
                    "datum.bin >= 0.5",
                    if_false={"value": "#2196F3"},
                    if_true={"value": "#FB8C00"},
                ),
            )
            .configure(background="transparent")
            .properties(width=self.width, height=75)
        )
        return chart

    def __panel__(self):
        return self.layout
