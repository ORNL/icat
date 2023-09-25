"""Data management class, this is another visual component, but also manages
all of the sample data and so on for the model.
"""

# Responsibilities:
# * Manages the current sample/sample history
# * Manages interaction with model as far as setting the training data and so on.
# * Tabulator and search and "tabs" component

from collections.abc import Callable
from typing import Optional

import ipyvuetify as v
import ipywidgets as ipw
import pandas as pd
import panel as pn
import param

import icat
from icat.anchors import DictionaryAnchor
from icat.item import ItemViewer
from icat.table import TableContentsTemplate
from icat.utils import _kill_param_auto_docstring, add_highlights

_kill_param_auto_docstring()


class DataManager(pn.viewable.Viewer):
    """A model's container and viewer component for labelling individual data points.

    This manages the current set of sample points and holds the currently explored
    dataset. It provides the Panel component for interacting with this dataset, in terms
    of filtering/searching through it, providing multiple different "view" sets, and

    Args:
        data (pd.DataFrame): The initial dataset to use.
        text_col (str): The column of the data containing the text to explore.
        model (Model): The parent model instance.
        width (int): The width of the data manager viewer table.
        height (int): The height of the data manager viewer table.
        default_sample_size (int): The number of points to randomly sample.
    """

    sample_indices = param.List([])
    """The row indices from the active_data in the current sample set."""

    selected_indices = param.List([])
    """The row indices from the active_data lasso-selected by the user."""

    update_trigger = param.Event()

    current_data_tab = param.String("Sample")
    search_value = param.String("")

    # the min/max prediction values from the histogram range slider
    pred_min = param.Number(0.0)
    pred_max = param.Number(1.0)

    # NOTE: since we're not yet treating width/height as params, you can't actually
    # change the size of the datamanager after its creation, which should be addressed
    # at some point. Note that for this to work, we'll probably need to still keep
    # the width/height as explicit init params, and then update the class params for
    # them _after_ we call super init (see note on that below). Then we'll need event
    # handlers to propagate those changes on down to tabledisplay and table contents.

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        text_col: str | None = None,
        model: Optional["icat.model.Model"] = None,
        width: int = 700,
        height: int = 800,
        default_sample_size: int = 100,
        **params,
    ):
        self.active_data = None
        """The current active dataset the user is exploring with the model."""
        self.filtered_df = None
        """The data currently displayed after the relevant filters are applied."""
        self.model = model

        self.width = width
        self.height = height

        self.text_col: str = text_col
        self.label_col: str = "_label"
        self.prediction_col: str = "_pred"

        self.item_viewer: ItemViewer = ItemViewer(height=height, width=width, data=self)
        self.item_viewer.on_label_changed(self._handle_label_changed)

        self.data_tab_list = ["Sample", "Labeled", "Interesting", "Selected", "All"]
        self.data_tabs = v.Tabs(
            v_model=0,
            fixed_tabs=True,
            dense=True,
            height=35,
            children=[v.Tab(children=[tab]) for tab in self.data_tab_list],
        )
        self.data_tabs.on_event("change", self._handle_ipv_tab_changed)

        self.search_box = v.TextField(
            v_model="",
            dense=True,
            append_icon="mdi-magnify",
            label="Search (use 'ID:X' to search index)",
            clearable=True,
            clear_icon="mdi-close",
            style_="padding-top: 7px",
        )
        self.search_box.on_event("keyup", self._handle_ipv_search_changed)
        self.search_box.on_event("click:clear", self._handle_ipv_search_cleared)

        self.search_add_new_btn = v.Btn(
            small=True,
            children=[v.Icon(children=["mdi-plus"]), "new"],
            style_="margin-top: 15px; margin-left: 5px;",
            v_on="add_tooltip.on",
        )
        self.search_add_new_btn.on_event(
            "click", self._handle_ipv_search_add_new_btn_click
        )
        self.search_add_sel_btn = v.Btn(
            small=True,
            children=[v.Icon(children=["mdi-plus"]), "sel"],
            style_="margin-top: 15px; margin-left: 5px;",
            v_on="sel_tooltip.on",
        )
        self.search_add_sel_btn.on_event(
            "click", self._handle_ipv_search_add_sel_btn_click
        )

        self.search_add_new_tooltip = v.Tooltip(
            top=True,
            open_delay=500,
            v_slots=[
                {
                    "name": "activator",
                    "variable": "add_tooltip",
                    "children": self.search_add_new_btn,
                }
            ],
            children=["Add search text to a new dictionary anchor."],
        )
        self.search_add_sel_tooltip = v.Tooltip(
            top=True,
            open_delay=500,
            v_slots=[
                {
                    "name": "activator",
                    "variable": "sel_tooltip",
                    "children": self.search_add_sel_btn,
                }
            ],
            children=["Add search text to currently selected dictionary anchor."],
        )

        self.table = TableContentsTemplate(
            headers=[
                {
                    "text": self.text_col,
                    "value": "text",
                    "width": f"{self.width-95-60-36}px",
                },
                {"text": "id", "value": "index", "width": "60px"},
                {"text": "Actions", "value": "actions", "width": "95px"},
            ],
            width=width,
            height=height - 240,
        )
        self.table.on_update_options(self._set_page_contents)
        self.table.on_apply_label(self._handle_label_changed)
        self.table.on_select_point(self.fire_on_row_selected)
        self.table.on_select_point(self._handle_row_selected)
        self.table.on_add_example(self._handle_example_added)
        self.table.on_add_to_sample(self._handle_sample_added)

        if self.model is not None:
            self.model.anchor_list.on_default_example_anchor_type_changed(
                self._handle_default_example_anchor_type_changed
            )
            # in case the anchor list has already been initialized with anchor types,
            # make sure to grab the existing anchor type as needed.
            if "ref" in self.model.anchor_list.default_example_anchor_type_dict:
                self._handle_default_example_anchor_type_changed(
                    self.model.anchor_list.default_example_anchor_type_dict
                )

        self.filtered_df = None

        self.label_all_i_btn = v.Btn(
            children=["All interesting"],
            small=True,
            class_="orange darken-1",
            style_="margin-left: 2px; margin-right: 2px;",
        )
        self.label_all_u_btn = v.Btn(
            children=["All uninteresting"],
            small=True,
            class_="blue darken-1",
            style_="margin-left: 2px; margin-right: 2px;",
        )
        self.label_all_i_btn.on_event("click", self._handle_ipv_label_all_i_btn_click)
        self.label_all_u_btn.on_event("click", self._handle_ipv_label_all_u_btn_click)
        self.label_all_row = v.Row(
            children=[self.label_all_u_btn, self.label_all_i_btn],
            style_="display: none; text-align: right; margin-right: 5px;",
        )

        # NOTE: ipywidgets_bokeh is a bit broken right now and I'm not yet able
        # to get pn.ipywidget to work correctly - so panel components can only
        # be at the highest level right now, I can't have ipywidgets with panel
        # components as children. To get around this, I have both a self.layout
        # (for panel) and a self.widget (for a roughly equivalent ipywidget version.)
        # This allows a user to still get a quick interactive component, e.g. by
        # running `model.data`, and for the more comprehensive model interactiveview
        # to get a ipywidgets friendly `model.data.widget`. In principle, once bokeh3,
        # ipywidgets8, and ipywidgets_bokeh are all playing nice, this won't be necessary
        # anymore.
        data_layout_stack = v.Container(
            fluid=True,
            children=[
                self.data_tabs,
                v.Row(
                    children=[
                        self.search_box,
                        self.search_add_new_tooltip,
                        self.search_add_sel_tooltip,
                    ],
                    style_="padding-left: 10px; padding-right: 10px;",
                ),
                self.label_all_row,
                self.table,
            ],
            height=height,
            width=width,
        )

        self.resample_btn = v.Btn(children=["Resample"])
        self.resample_btn.on_event("click", self._handle_ipv_resample_btn_click)
        self.sample_size_txt = v.TextField(
            v_model=str(default_sample_size), label="Sample size"
        )
        self.sampling_controls = v.Container(
            children=[
                v.Row(
                    children=[
                        self.sample_size_txt,
                        self.resample_btn,
                    ],
                    style_="padding-left: 10px; padding-right: 10px;",
                ),
            ],
            height=height,
            style_=f"width: {width}px",
        )

        self.tabs_component = v.Tabs(
            v_model=0,
            height=35,
            background_color="primary",
            style_=f"width: {width}px",
            children=[
                v.Tab(children=["Data"]),
                v.Tab(children=["Item"]),
                v.Tab(children=["Sampling"]),
            ],
        )
        self.tabs_items_component = v.TabsItems(
            v_model=0,
            style_=f"width: {width}px",
            children=[
                v.TabItem(children=[data_layout_stack]),
                v.TabItem(children=[self.item_viewer.widget]),
                v.TabItem(children=[self.sampling_controls]),
            ],
        )
        ipw.jslink(
            (self.tabs_component, "v_model"), (self.tabs_items_component, "v_model")
        )

        layout_stack = v.Container(
            children=[self.tabs_component, self.tabs_items_component],
            style_=f"padding: 0px; width: {width}px",
        )
        self.layout = pn.Column(
            layout_stack, height=height, width=width, styles={"padding": "0px"}
        )
        self.widget = v.Container(
            children=[layout_stack],
            height=height,
            style_=f"padding: 0px; width: {width}px",
        )

        self._data_label_callbacks: list[Callable] = []
        self._row_selected_callbacks: list[Callable] = []
        self._sample_changed_callbacks: list[Callable] = []

        super().__init__(**params)  # required for panel components
        # Note that no widgets can be declared _after_ the above, or their values won't be
        # considered parameters. But, calling this also resets any changes to local params,
        # so any init code that changes param values will need to go _after_ this.

        if data is not None:
            self.set_data(data)

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def _handle_ipv_search_add_new_btn_click(self, widget, event, data):
        """Event handler for when the add search box text to new anchor button is clicked."""
        if self.search_value != "":
            new_anchor = DictionaryAnchor(keywords=[self.search_value])
            self.model.add_anchor(new_anchor)
            self.model.view.anchorviz.selectedAnchorID = new_anchor.name
        # self.search_box.v_model = ""
        # self.search_value = ""

    def _handle_ipv_search_add_sel_btn_click(self, widget, event, data):
        """Event handler for when the add search box text to new anchor button is clicked."""
        if self.search_value != "" and self.model is not None:
            anchor = self.model.anchor_list.get_anchor_by_panel_id(
                self.model.view.anchorviz.selectedAnchorID
            )
            anchor.keywords = [*anchor.keywords, self.search_value]

    def _handle_ipv_label_all_i_btn_click(self, widget, event, data):
        indices = self.filtered_df.index.tolist()
        labels = [1] * len(indices)
        self._handle_label_changed(indices, labels)

    def _handle_ipv_label_all_u_btn_click(self, widget, event, data):
        indices = self.filtered_df.index.tolist()
        labels = [0] * len(indices)
        self._handle_label_changed(indices, labels)

    # TODO: coupling: directly calling refresh data on the model view. Instead, we
    # could have a "sample_changed" event that view listens to.
    def _handle_ipv_resample_btn_click(self, widget, event, data):
        self.set_random_sample()

    def _handle_ipv_tab_changed(self, widget, event, data: int):
        """Event handler for the vuetify tabs change. This changes the current_data_tab
        param, which will automatically trigger the apply_filter."""
        self.table.options["page"] = 1
        # BUG: https://github.com/ORNL/icat/issues/3
        # Sometimes when clicking on the first tab, the data being sent is a blank dictionary
        # {} instead of the integer 0. Bypassing this by directly setting the tab based on
        # data_tabs v_model which does correctly get set to 0. This is possibly an issue within
        # ipyvuetify itself?
        # self.current_data_tab = self.data_tab_list[data]
        self.current_data_tab = self.data_tab_list[self.data_tabs.v_model]
        if self.current_data_tab == "Selected":
            self.label_all_row.style_ = (
                "display: block; text-align: right; margin-right: 5px;"
            )
        else:
            self.label_all_row.style_ = (
                "display: none; text-align: right; margin-right: 5px;"
            )

    def _handle_ipv_search_changed(self, widget, event, data):
        """Event handler for the vuetify search box change. This changes the search_value
        param, which will automatically trigger the apply_filter."""
        self.search_value = widget.v_model if widget.v_model is not None else ""

    def _handle_ipv_search_cleared(self, widget, event, data):
        """Event handler for the vuetify search box "x" button pressed, resetting search field."""
        self.search_value = ""
        self.search_box.success = False
        self.search_box.error = False
        self.search_box.label = "Search (use 'ID:X' to search index)"

    def _handle_row_selected(self, point_id):
        """Event handler from table row select."""
        self.item_viewer.populate(point_id)
        self.tabs_component.v_model = 1

    def _handle_example_added(self, point_id):
        """Event handler for when the 'example' button is clicked."""
        if self.model is not None:
            new_anchor = self.model.anchor_list.default_example_anchor_type_dict["ref"](
                text_col=self.text_col, container=self.model.anchor_list
            )

            # TODO: an example of where it would be cleaner to just be able to pass in ID's
            # directly on init
            new_anchor.reference_texts = [self.active_data.loc[point_id, self.text_col]]
            # TODO: coupling: honestly I don't have a ton of heartburn from this one.
            # Arguably model should itself have an add_anchor function (which would just
            # be calling the anchor_list one, but law of demeter) which would make this
            # marginally cleaner and make model's interface nicer as well.
            self.model.anchor_list.add_anchor(new_anchor)

    def _handle_sample_added(self, point_id):
        """Event handler for when the 'sample' button is clicked."""
        self.sample_indices = [*self.sample_indices, point_id]

    def _handle_label_changed(self, index: int | list[int], new_label: int | list[int]):
        if type(index) == list:
            for i in range(len(index)):
                self.active_data.at[index[i], self.label_col] = new_label[i]
        else:
            self.active_data.at[index, self.label_col] = new_label
        self.fire_on_data_labeled(index, new_label)
        self._apply_filters()  # ....this doesn't feel good, but some filters depend on labeling?

    def _handle_default_example_anchor_type_changed(
        self, anchor_type_dict: dict[str, any]
    ):
        if "name" in anchor_type_dict:
            self.table.example_btn_color = anchor_type_dict["color"]
            self.table.example_type_name = anchor_type_dict["name"]
        else:
            # if we get here, a blank dictionary was likely passed, which means the example
            # anchor type was removed.
            self.table.example_btn_color = ""
            self.table.example_type_name = "similarity"

    # ============================================================
    # EVENT SPAWNERS
    # ============================================================

    # TODO: hook this up to model view instead of manually calling in resmaple button click handler
    def on_sample_changed(self, callback: Callable):
        """Register a callback function for the "sample changed" event.

        Callbacks for this event should take a single parameter which is
        the new list of sample indices
        """
        self._sample_changed_callbacks.append(callback)

    def on_data_labeled(self, callback: Callable):
        """Register a callback function for the "data label changed" event.

        Note that depending on how it's fired, this can either apply to a single datapoint
        being labeled, or a set of points.

        Callbacks for this event should take two parameters:
        * index (int | list[int])
        * label (int | list[int])

        If index is a list, that means multiple points are being labeled simultaneously.
        """
        self._data_label_callbacks.append(callback)

    def on_row_selected(self, callback: Callable):
        """Register a callback function for the "row clicked" event.

        Callbacks for this event should take the index as a parameter.
        """
        self._row_selected_callbacks.append(callback)

    @param.depends("sample_indices", watch=True)
    def fire_on_sample_changed(self):
        for callback in self._sample_changed_callbacks:
            callback(self.sample_indices)

    def fire_on_data_labeled(self, index: int | list[int], label: int | list[int]):
        for callback in self._data_label_callbacks:
            callback(index, label)

    def fire_on_row_selected(self, index: int):
        for callback in self._row_selected_callbacks:
            callback(index)

    # ============================================================
    # INTERNAL FUNCTIONS
    # ============================================================

    def __panel__(self):
        return self.layout

    @param.depends(
        "current_data_tab",
        "sample_indices",
        "selected_indices",
        "search_value",
        "update_trigger",
        "pred_min",
        "pred_max",
        watch=True,
    )
    def _apply_filters(self):
        self.data_tabs.v_model = self.data_tab_list.index(self.current_data_tab)
        df = self.active_data
        df = self._current_tab_filter(
            df, self.current_data_tab, self.sample_indices, self.selected_indices
        )
        df = self._search_box_filter(df, self.search_value, self.text_col)
        df = self._prediction_range_filter(df)
        self.filtered_df = df

        self.table.total_length = self.filtered_df.shape[0]
        # self.table.options["page"] = 1  # TODO: necessary?
        self._set_page_contents(self.table.options)

    def _set_page_contents(self, options):
        """Retrieves rows from the filtered dataframe based on current table options
        and sets them."""

        if self.filtered_df is None:
            # I think this function gets triggered on table widget init, which isn't
            # necessary, so just ignore
            return

        if self.text_col is None:
            return

        df = self.filtered_df
        page_num = 1
        if "page" in options:
            page_num = options["page"]
        count = 10
        if "itemsPerPage" in options:
            count = options["itemsPerPage"]

        # sort the dataframe if requested (has to be done here since data external to JS)
        if "sortBy" in options and len(options["sortBy"]) > 0:
            if options["sortBy"][0] == "index":
                df = df.sort_index(ascending=(not options["sortDesc"][0]))
            else:
                df = df.sort_values(
                    by=options["sortBy"][0], ascending=(not options["sortDesc"][0])
                )

        # calculate the range in the dataframe based on current page
        total_len = df.shape[0]
        end_index = page_num * count
        start_index = end_index - count
        if end_index > total_len:
            end_index = total_len
        df_rows = df.iloc[start_index:end_index, :]

        # create a list of dictionaries for each row in range to assign to the table items
        rows = []
        for index, row in df_rows.iterrows():
            # handle keyword highlighting
            text = row[self.text_col]
            if self.model is not None:
                # TODO: coupling: I don't have a huge issue with this and I don't know that
                # there's a better way of doing it anyway. We _have_ to know about the anchors
                # here somehow
                kw_regex = self.model.anchor_list.highlight_regex()
                text = add_highlights(text, kw_regex)
                text = add_highlights(text, f"({self.search_value})", "white")

            # set the color of the text based on the prediction
            if self.prediction_col in row.keys():
                if row[self.prediction_col] > 0.5:
                    color = "orange"
                else:
                    color = "blue"
                text = f"<span class='{color}--text darken-1'>{text}</span>"
            else:
                text = f"<span>{text}</span>"

            # if a point has been explicitly labeled, add a little colored text indicator
            if self.label_col in row.keys():
                if row[self.label_col] == -1:
                    labeled = ""
                else:
                    if row[self.label_col] == 1:
                        labeled = "orange"
                    elif row[self.label_col] == 0:
                        labeled = "blue"
                    labeled = f"<span class='{labeled}--text darken-1'>Labeled</span>"

            rows.append(
                dict(
                    id=index,
                    text=text,
                    labeled=labeled,
                    in_sample=(index in self.sample_indices),
                )
            )
        self.table.items = rows

    def _search_box_filter(
        self,
        df: pd.DataFrame,
        pattern: str,
        column: str,
    ) -> pd.DataFrame:
        """This function searches for a given string in code:`pattern` and applies it to the code:`column` within the
        code:`df`.

        If the search is "ID:", we directly search the index column instead.
        """
        if not pattern:
            return df

        # check for an index search
        if pattern.startswith("ID:"):
            requested_index = str(pattern[3:])
            if requested_index.isnumeric():
                self.search_box.label = "Search (index search mode)"
                if int(requested_index) in df.index:
                    self.search_box.success = True
                    self.search_box.error = False
                    return df.loc[[int(requested_index)]]
                else:
                    self.search_box.success = False
                    self.search_box.error = True
                    return pd.DataFrame()

        self.search_box.label = "Search (use 'ID:X' to search index)"
        self.search_box.success = False
        self.search_box.error = False
        return df[df[column].str.contains(pattern, case=False)]

    # TODO: either make this static and add prediction col,
    # or embrace that it's a self filter and don't support passing min/max/take it from the self params
    def _prediction_range_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out rows that don't have a predicted value within the specified range."""

        if self.prediction_col not in df:
            return df

        return df[
            (df[self.prediction_col] >= self.pred_min)
            & (df[self.prediction_col] <= self.pred_max)
        ]

    def _current_tab_filter(
        self,
        df: pd.DataFrame,
        filter_name: str = None,
        current_sample: list[int] = None,
        selected_indices: list[int] = None,
    ) -> pd.DataFrame:
        if df is None:
            return None

        if filter_name is None:
            return df

        # determine filter method based on the tab name that was passed
        if filter_name == "Sample":
            if current_sample is not None:
                return df.loc[current_sample, :]
            else:
                return df.loc[[], []]
        elif filter_name == "Labeled":
            if self.label_col in df.columns:
                return df[(df[self.label_col] != -1) & (df[self.label_col] != pd.NA)]
            else:
                return df.loc[[], []]
        elif filter_name == "Interesting":
            # only get rows where the prediction is "interesting" (>0.5)
            if self.prediction_col in df.columns:
                return df[
                    (df[self.prediction_col] != -1)
                    & (df[self.prediction_col] != pd.NA)
                    & (df[self.prediction_col] > 0.5)
                ]
            else:
                return df.loc[[], []]
        elif filter_name == "Selected":
            if selected_indices is not None:
                return df.loc[selected_indices, :]
            else:
                return df.loc[[], []]
        else:
            return df

    # ============================================================
    # PUBLIC FUNCTIONS
    # ============================================================

    def apply_label(self, index: int | list[int], label: int | list[int]):
        """Provide the label(s) for the specified index/indices.

        Args:
            index (int | list[int]): Either a single index, or a list of indices.
            label (int | list[int]): Either the single label to apply or a list of corresponding labels \
                for the provided indices. 1 is "interesting", 0 is "uninteresting". If a -1 is provided, \
                this resets or "unlabels", removing it from the container model's training set.
        """
        self._handle_label_changed(index, label)

    def set_data(self, data: pd.DataFrame):
        """Replace the current active data with the data specified.

        Note that this won't wipe out the existing training data for the model, model.training_data
        is a separate data frame that's built up as labels are applied to various datasets.

        Args:
            data (pd.DataFrame): The dataset to use as the current active data.
        """
        # TODO: option to allow "Keeping indices/samples", for example if you were
        # doing a bunch of table updates, but no actual row additions/removals/index changes
        self.active_data = data.copy()
        # self.table.value = data

        # add any missing columns
        if self.label_col not in self.active_data:
            self.active_data[self.label_col] = -1

        self.set_random_sample()
        # TODO: seems weird to handle this here
        self._apply_filters()

    def set_random_sample(self):
        """Randomly choose 100 indices to use for the anchorviz sample."""

        sample_size = int(self.sample_size_txt.v_model)

        if len(self.active_data) > sample_size:
            self.sample_indices = list(self.active_data.sample(sample_size).index)
        else:
            self.sample_indices = list(self.active_data.index)
