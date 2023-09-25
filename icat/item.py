"""Class for interface to view and interact with a single sample/instance point."""

from collections.abc import Callable

import ipyvuetify as v
import panel as pn
import traitlets

import icat
from icat.utils import _kill_param_auto_docstring, add_highlights

_kill_param_auto_docstring()


class HtmlContents(v.VuetifyTemplate):
    """A vuetify template to inject text HTML into a div, this is necessary
    because ipyvuetify does not appear to have a way to set the v-html attribute"""

    content = traitlets.Unicode("").tag(sync=True)

    @traitlets.default("template")
    def _template(self):
        return """
            <div v-html="content" />
        """


class ItemViewer(pn.viewable.Viewer):
    """Interface for viewing and labeling a single item, useful for looking
    at more than just a snippet of the full text.

    Args:
        index (int): The row index of the entry in the dataframe to view.
        width (int): The width of the rendered card.
        height (int): The height of the rendered card.
        data (DataManager): The parent data manager to pull the item from.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 700,
        height: int = 800,
        # TODO: coupling: same as with histograms, since there's no functions being called
        # through parent data, I think this is a low priority change
        data: "icat.data.DataManager" = None,
        **params,
    ):
        super().__init__(**params)  # required for panel components

        self.width = width
        self.height = height
        self.data = data

        self.index = index

        self.contents = HtmlContents(content="<p>No instance selected.</p>")
        self.interesting_button = v.Btn(
            style_="padding: 5px; margin: 5px;",
            color="orange",
            children=["Interesting"],
        )
        self.interesting_button.on_event(
            "click", self._handle_ipv_interesting_btn_clicked
        )
        self.uninteresting_button = v.Btn(
            style_="padding: 5px; margin: 5px;",
            color="blue",
            children=["Uninteresting"],
        )
        self.uninteresting_button.on_event(
            "click", self._handle_ipv_uninteresting_btn_clicked
        )
        self.current_label = v.Html(
            style_="padding: 5px;", tag="p", children=["Not labeled"]
        )
        self.current_prediction = v.Html(
            style_="padding: 5px;", tag="p", children=["No prediction"]
        )
        self.index_display = v.Html(tag="h3", style_="margin: 5px", children=[])

        stack = v.Container(
            children=[
                v.Col(
                    children=[
                        v.Row(children=[self.index_display]),
                        v.Row(
                            children=[
                                self.uninteresting_button,
                                self.interesting_button,
                                self.current_label,
                                self.current_prediction,
                            ]
                        ),
                        v.Row(
                            children=[
                                v.Card(
                                    children=[
                                        v.CardText(
                                            style_=f"height: {height-220}px; overflow-y: scroll;",
                                            children=[self.contents],
                                        )
                                    ]
                                )
                            ]
                        ),
                    ]
                )
            ]
        )

        self.widget = v.Container(height=height, width=width, children=[stack])
        self.layout = pn.Column(stack, width=self.width, height=self.height)

        self._label_changed_handlers: list[Callable] = []

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def _handle_ipv_interesting_btn_clicked(self, widget, event, data):
        self.fire_on_label_changed(1)

    def _handle_ipv_uninteresting_btn_clicked(self, widget, event, data):
        self.fire_on_label_changed(0)

    # ============================================================
    # EVENT SPAWNERS
    # ============================================================

    def on_label_changed(self, callback: Callable):
        """Register a callback function for the "label changed" event.

        Callbacks for this event should take two parameters:
        * index of labeled point (int)
        * label value (int)
        """
        self._label_changed_handlers.append(callback)

    def fire_on_label_changed(self, label: int):
        for callback in self._label_changed_handlers:
            callback(self.index, label)

    # ============================================================
    # INTERNAL FUNCTIONS
    # ============================================================

    def __panel__(self):
        return self.layout

    # ============================================================
    # PUBLIC FUNCTIONS
    # ============================================================

    # TODO: coupling, datamanger could have a populate_instance function
    # and this one takes the color, pred value, and text
    # Another thought regarding highlights is that the model itself could have
    # a "highlight_features" function that takes care of getting the regex from
    # anchorlist and injecting the span styles.
    def populate(self, index: int):
        """Fill or update all of the fields for the given index. This
        should be called anytime the model updates, or when the user
        clicks/requests to view a new instance.

        Args:
            index (int): The row index of the item to display from parent \
                DataManager's active_data.
        """
        if self.data.active_data is None:
            return

        self.index = index
        self.index_display.children = [f"ID:{index}"]
        row = self.data.active_data.iloc[index]

        # highlight text
        text = row[self.data.text_col]
        if self.data.model is not None:
            kw_regex = self.data.model.anchor_list.highlight_regex()
            text = add_highlights(text, kw_regex)
            text = add_highlights(text, f"({self.data.search_value})", "white")

        self.contents.content = text

        # set color of prediction value
        if self.data.prediction_col in row and row[self.data.prediction_col] != -1:
            self.current_prediction.children = [
                f"Prediction: {row[self.data.prediction_col]}"
            ]
            if row[self.data.prediction_col] > 0.5:
                color = "orange"
            else:
                color = "blue"
            self.current_prediction.class_ = f"{color}--text darken-1"

        # set the "labeled" label and color if applicable
        if self.data.label_col in row and row[self.data.label_col] != -1:
            self.current_label.children = ["Labeled"]
            if row[self.data.label_col] >= 0.5:
                color = "orange"
            else:
                color = "blue"
            self.current_label.class_ = f"{color}--text darken-1"
        else:
            self.current_label.children = ["Not labeled"]
            self.current_label.class_ = ""
