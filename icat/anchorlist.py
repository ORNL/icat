"""The AnchorList component class file, these manage a model's current set of anchors."""

import json
import os
import pickle
from collections.abc import Callable
from functools import partial

import ipyvuetify as v
import ipywidgets as ipw
import numpy as np
import pandas as pd
import panel as pn
import param
import traitlets
from sklearn.feature_extraction.text import TfidfVectorizer

from icat.anchors import Anchor, DictionaryAnchor, SimilarityAnchorBase, TFIDFAnchor
from icat.utils import _kill_param_auto_docstring, vue_template_path

_kill_param_auto_docstring()


ANCHOR_COLOR_PALLETE = [
    [
        "#2ABD7F",  # pleasant pale green
        "#42803F",  # dark green
    ],
    [
        "#8C27C1",  # dark purple
        "#5B3D7B",  # darker purple
    ],
    [
        "#2EA5A5",  # mid pale blue
        "#424E80",  # dark washed blue
    ],
    [
        "#BFA74C",  # gold
        "#6E9613",  # olive green
    ],
    [
        "#DA707C",  # salmon
        "#A75B13",  # burnt orange
    ],
    [
        "#CA3296",  # pink
        "#881515",  # red
    ],
    [
        "#8D6E63",  # brown
        "#777777",  # grey
    ],
]


class AnchorListTemplate(v.VuetifyTemplate):
    """The ipyvuetify contents of the anchorlist table. This handles all the special considerations
    like the slot for the expanded rows containing the anchor widgets, and the coverage v-html etc.
    """

    template_file = vue_template_path("anchors-table.vue")

    items = traitlets.List().tag(sync=True, **ipw.widget_serialization)
    # TODO: headers shouldn't actually change, this is fixed
    headers = traitlets.List(
        [
            {"text": "Anchor Name", "value": "anchor_name"},
            {"text": "Cov", "value": "coverage"},
            {"text": "% Neg", "value": "pct_negative"},
            {"text": "% Pos", "value": "pct_positive"},
            {"text": "Viz", "value": "in_viz", "width": 10, "sortable": False},
            {"text": "Model", "value": "in_model", "width": 10, "sortable": False},
            {"text": "Delete", "value": "delete", "width": 10, "sortable": False},
        ]
    ).tag(sync=True)
    # TODO: height and width don't seem to be used?
    height = traitlets.Int(700).tag(sync=True)
    width = traitlets.Int(150).tag(sync=True)
    expanded = traitlets.List([]).tag(sync=True)
    processing = traitlets.Bool(False).tag(sync=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._anchor_removal_callbacks: list[Callable] = []

    def on_anchor_removal(self, callback: Callable):
        """Register a callback for when the delete button for a row is clicked.

        Callbacks should take a single parameter that is the panel name of the anchor.
        """
        self._anchor_removal_callbacks.append(callback)

    def vue_deleteAnchor(self, name: str):
        """Ipyvuetify event handler for the x-button click."""
        for callback in self._anchor_removal_callbacks:
            callback(name)

    def _set_anchor_processing(self, name: str, processing: bool):
        """Add or remove a loading spinner on the delete button to indicate anchor activity.

        Args:
            name (str): The panel component name id.
            processing (bool): Whether the anchor is busy or not.
        """
        new_items = []
        for item in self.items:
            if item["name"] == name:
                # we have to create a new item with the same fields in order for the traitlet
                # to recognize this is actually a new list
                new_item = dict(**item)
                new_item["processing"] = processing
                new_items.append(new_item)
            else:
                new_items.append(item)
        self.items = new_items


class AnchorList(pn.viewable.Viewer):
    """A model's list tracking and managing a collection of anchors.

    This is what handles creating features for a dataset. This class is also a
    visual component for interacting with and modifying those anchors in a table
    format and is used as part of the greater model interactive view.

    Args:
        model: The parent model.
        table_width (int): Static width of the visual component table.
        table_height (int): Static height of the visual component table. \
            (Currently unused)
        anchor_types (list[type | dict[str, any]]): The anchor types to start the \
            interface with. This list can contain a combination of types and \
            dictionaries with keys ``ref`` (containing the type), ``name`` (display \
            name) and ``color`` (the color to render anchors of this type with.) \
            If left None (the default), will add ``DictionaryAnchor`` and ``TFIDFAnchor``.
    """

    anchors = param.List([])
    """The ``param`` list of anchors.

    Important:
        Care must be taken when directly setting this variable. Lists are mutable, and
        panel/param do not detect inner changes to mutable objects. This includes things like
        ``anchors.append(my_anchor)`` and ``anchors[1] = some_anchor``. Use the various
        functions on this class such as ``add_anchor`` and so on. Notably, entire list replacements,
        e.g. ``anchors = [my_anchor, some_anchor]``, work as expected.
    """

    # TODO: I don't love that this dictionary is exactly replicating the info of one item
    # in possible_anchor_types, could potentially lead to desyncing issues if not handled carefully
    default_example_anchor_type_dict = param.Dict({})
    """The dictionary of the anchor type information for the current default example anchor type.
    This should contain keys ``ref`` (type), ``name``, and ``color``."""

    possible_anchor_types = param.List([])
    """The list of registered anchor types within the interface, each entry is a dictionary
    containing ``ref``, ``name``, and ``color``. Note that this list should not be manually
    altered, as sub parameter changes won't be picked up by the interface. Use ``add_anchor_type``,
    ``modify_anchor_type``, and ``remove_anchor_type``."""

    # TODO: coupling: model is used to retrieve model.data.active_data
    # NOTE: one way around this would be to have a "reference_data" property,
    # where the setter (which could be hooked up through a model event) re-
    # computes relevant tfidf vals
    def __init__(
        self,
        model=None,
        table_width: int = 700,
        table_height: int = 150,
        anchor_types: list[type | dict[str, any]] = None,
        **params,
    ):
        super().__init__(**params)  # required for panel components

        if anchor_types is None:
            anchor_types = [
                DictionaryAnchor,
                {"ref": TFIDFAnchor, "color": ANCHOR_COLOR_PALLETE[0][1]},
            ]

        self.coverage_info = {}
        """Dictionary associating panel id of anchor with dictionary of 'coverage', 'pct_positive',
        and 'pct_negative' stats."""

        self.expand_toggle = v.Btn(
            icon=True,
            fab=True,
            x_small=True,
            plain=True,
            children=[v.Icon(children=["mdi-expand-all"])],
            visible=False,
            style_="position: absolute; margin-top: 28px; margin-left: 10px;",
            # jaaaaaaaaank, but it works and this way I don't have to re-implement the entire
            # header row slot in the anchorlisttemplate, which would have to include the sorting
            # actions and icons etc.
            v_on="tooltip.on",
        )
        self.expand_toggle.on_event("click", self._handle_ipv_expand_toggle_click)

        self.expand_toggle_tooltip = v.Tooltip(
            top=True,
            open_delay=500,
            v_slots=[
                {
                    "name": "activator",
                    "variable": "tooltip",
                    "children": self.expand_toggle,
                }
            ],
            children=["Collapse/expand all."],
        )

        self.table = AnchorListTemplate()
        self.table.on_anchor_removal(self._handle_table_anchor_deleted)

        self.anchor_buttons = v.Row(
            children=[
                self.expand_toggle_tooltip,
            ]
        )
        self.anchors_layout = v.Col(
            children=[
                self.anchor_buttons,
                self.table,
            ],
            style_=f"padding-left: 0; padding-right: 0px; padding-bottom: 0px; width: {table_width}px;",
        )
        """The full component layout for panel to display."""

        # --- should move out to anchortypes.py ---
        self.anchor_types_layout = v.Col(style_=f"width: {table_width}px")

        # --- should move out to anchorsettings.py ---
        self.example_anchor_types_dropdown = v.Select(
            label="Default example anchor type", items=[]
        )
        self.example_anchor_types_dropdown.on_event(
            "change", self._handle_ipv_default_example_anchor_type_changed
        )
        self.anchor_settings_layout = v.Col(
            children=[self.example_anchor_types_dropdown],
            style_=f"width: {table_width}px",
        )
        # ---/---

        self.tabs_component = v.Tabs(
            v_model=0,
            height=35,
            background_color="primary",
            width=table_width,
            children=[
                v.Tab(children=["Anchors"]),
                v.Tab(children=["Anchor Types"]),
                v.Tab(children=["Settings"]),
            ],
        )
        self.tabs_items_component = v.TabsItems(
            v_model=0,
            width=table_width,
            children=[
                v.TabItem(children=[self.anchors_layout]),
                v.TabItem(children=[self.anchor_types_layout]),
                v.TabItem(children=[self.anchor_settings_layout]),
            ],
        )
        ipw.jslink(
            (self.tabs_component, "v_model"), (self.tabs_items_component, "v_model")
        )

        self.layout_stack = v.Container(
            children=[self.tabs_component, self.tabs_items_component],
            width=table_width,
            style_="padding: 0px;",
        )
        self.layout = pn.Column(
            self.layout_stack,
            width=table_width,
            styles={"padding": "0px"},
        )
        self.widget = v.Container(
            children=[self.layout_stack], width=table_width, style_="padding: 0px;"
        )

        self.cache: dict[str, any] = {}
        """This cache gets pickled on save, useful for anchors to store results of
        processing-intensive tasks to reduce featurizing time."""

        # TODO: can remove (should go through cache)
        self.tfidf_vectorizer = None
        self.tfidf_features = None

        self.model = model
        """The parent model these anchors apply to."""

        # callback collections
        self._anchor_changed_callbacks: list[Callable] = []
        self._anchor_added_callbacks: list[Callable] = []
        self._anchor_removed_callbacks: list[Callable] = []
        self._anchor_types_changed_callbacks: list[Callable] = []
        self._default_example_anchor_type_changed_callbacks: list[Callable] = []

        self.add_anchor_types(anchor_types)

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def _handle_table_anchor_deleted(self, name: str):
        """Event handler for the list table template delete button"""
        anchor = self.get_anchor_by_panel_id(name)
        self.remove_anchor(anchor)

    def _handle_ipv_expand_toggle_click(self, widget, event, data):
        # if any are expanded, set expanded to blank list
        if len(self.table.expanded) > 0:
            self.table.expanded = []
        else:
            self.table.expanded = [{"name": item["name"]} for item in self.table.items]

    def _handle_ipv_new_anchor_generic_click(self, widget, event, data, type_ref):
        # must be assigned with partial and assigning type_ref
        name = self.get_unique_anchor_name()
        self.add_anchor(type_ref(anchor_name=name))

    def _handle_ipv_new_anchor_type_add_click(
        self, widget, event, data, type_ref, text_field_ref: v.TextField
    ):
        # must be assigned with partial and assigning type_ref and text_field_ref
        self.add_anchor_type(type_ref, name=text_field_ref.v_model)

    def _handle_ipv_rm_anchor_type_btn_click(self, widget, event, data, type_ref: type):
        # must be assigned with partial assigning type_ref
        self.remove_anchor_type(type_ref)

    def _handle_ipv_color_picker_input(self, widget, event, data, anchor_type: type):
        # NOTE: need to set this up with a partial that specifies the anchor_type
        self.modify_anchor_type(anchor_type, "color", data)

    def _handle_ipv_anchor_type_name_changed(
        self, widget, event, data, anchor_type: type
    ):
        # NOTE: need to set this up with a partial that specifies the anchor_type
        self.modify_anchor_type(anchor_type, "name", widget.v_model)

    @param.depends("possible_anchor_types", watch=True)
    def _handle_pnl_possible_anchor_types_changed(self):
        new_anchor_buttons = []
        for anchor_type_dict in self.possible_anchor_types:
            new_button = v.Btn(
                children=[anchor_type_dict["name"]],
                small=True,
                color=anchor_type_dict["color"],
                style_="margin-left: 2px; margin-right: 2px",
                v_on="tooltip.on",
            )
            new_button.on_event(
                "click",
                partial(
                    self._handle_ipv_new_anchor_generic_click,
                    type_ref=anchor_type_dict["ref"],
                ),
            )

            button_tooltip = v.Tooltip(
                bottom=True,
                open_delay=500,
                max_width=400,
                v_slots=[
                    {
                        "name": "activator",
                        "variable": "tooltip",
                        "children": new_button,
                    }
                ],
                children=[anchor_type_dict["ref"].DESCRIPTION],
            )

            new_anchor_buttons.append(button_tooltip)
        self.anchor_buttons.children = [
            self.expand_toggle_tooltip,
            v.Html(
                tag="p",
                children=["New:"],
                style_="margin-bottom: 0px; margin-left: 18px; margin-top: 4px; margin-right: 3px;",
            ),
            *new_anchor_buttons,
        ]
        self.refresh_anchors_table()

    def _handle_ipv_default_example_anchor_type_changed(self, widget, event, data):
        self.default_example_anchor_type_dict = self.possible_anchor_types[data]
        self.fire_on_default_example_anchor_type_changed()

    # ============================================================
    # EVENT SPAWNERS
    # ============================================================

    def on_anchor_added(self, callback: Callable):
        """Register a callback function for the "anchor added" event.

        Callbacks for this event should take a single parameter which is the anchor that
        was added.
        """
        self._anchor_added_callbacks.append(callback)

    def on_anchor_changed(self, callback: Callable):
        """Register a callback function for the "anchor changed" event.

        Callbacks for this event should take three parameters:
        * Name (string) (this is the internal panel name, which we use as the anchor id.)
        * Property name (string) that's changing on the anchor.
        * Value that the property on the anchor was changed to.
        """
        self._anchor_changed_callbacks.append(callback)

    def on_anchor_removed(self, callback: Callable):
        """Register a callback function for the "anchor removal" event.

        Callbacks for this event should take a single parameter which is the anchor that
        was removed.
        """
        self._anchor_removed_callbacks.append(callback)

    def on_anchor_types_changed(self, callback: Callable):
        """Register a callback functino for the "anchor types changed" event.

        Callbacks for this event should take the list of dictionaries of type information,
        expect each dictionary to contain "name", "ref", and "color".
        """
        self._anchor_types_changed_callbacks.append(callback)

    def on_default_example_anchor_type_changed(self, callback: Callable):
        """Register a callback function for the "default example anchor changed
        event.

        Callbacks for this event should take the anchor type config dictionary, which
        contains "name", "ref", and "color".
        """
        self._default_example_anchor_type_changed_callbacks.append(callback)
        pass

    def fire_on_anchor_added(self, anchor: Anchor):
        """Trigger the event to notify that a new anchor was added.

        Args:
            anchor (icat.Anchor): the anchor that was added.
        """
        for callback in self._anchor_added_callbacks:
            callback(anchor)

    def fire_on_anchor_removed(self, anchor: Anchor):
        """Trigger the event to notify that an anchor was removed.

        Args:
            anchor (icat.Anchor): the anchor that was removed.
        """
        for callback in self._anchor_removed_callbacks:
            callback(anchor)

    def fire_on_anchor_changed(self, name: str, key: str, value):
        """Trigger the event to notify that a property on an anchor changed.

        Note that this is usually used for simply passing along the change directly
        from the anchor's individual on_anchor_changed events

        Args:
            name (str): the *internal name* that panel is using, which we're using as \
                the anchor id.
            key (str): the name of the property being changed.
            value (any): The value the property was changed to.
        """
        for callback in self._anchor_changed_callbacks:
            callback(name, key, value)

    def fire_on_anchor_types_changed(self):
        """Trigger the event to notify that the anchor types have been changed."""
        for callback in self._anchor_types_changed_callbacks:
            callback(self.possible_anchor_types)

    def fire_on_default_example_anchor_type_changed(self):
        """Trigger the event to notify that the default anchor type used for 'example'
        instances is changed."""
        for callback in self._default_example_anchor_type_changed_callbacks:
            callback(self.default_example_anchor_type_dict)

    # ============================================================
    # INTERNAL FUNCTIONS
    # ============================================================

    def __panel__(self):
        return self.layout

    def _get_row_contents(self, row):
        """This grabs the 'expanded' version of the requested row by index, which should
        return the visual components for interacting with that particular anchor.

        Note:
            this is a function we pass to tabulator, it handles calling this internally.
        """
        anchor = self.anchors[row["id"]]
        return anchor.row_view()

    # TODO: coupling: shouldn't need this once container is removed from anchors
    @param.depends("anchors", watch=True)
    def _inject_self(self):
        """Set the container of each anchor, this happens automatically whenever anchors is
        changed/set (NOTE: not mutated, this is why we need add_anchor to handle this separately)
        so that the user doesn't have to pass anchorlist to each anchor constructor.
        """
        for anchor in self.anchors:
            if self.fire_on_anchor_changed not in anchor._anchor_changed_callbacks:
                anchor.on_anchor_changed(self.fire_on_anchor_changed)

            # TODO: remove once tfidf anchor is handled
            if anchor.container is None or anchor.container != self:
                anchor.container = self

    @staticmethod
    def _l1_col_normalize(
        feature: pd.Series, reference_features: pd.DataFrame | None = None
    ):
        # NOTE: we're leaving this as the l1-norm for now because that is what was in
        # the old code, and it seemed to work. Also, it appears like this is what they
        # use in the AnchorViz paper, although it's not entirely clear what "j" is.
        # Note that in the RadViz paper, they mention "local" versus "global"
        # normalization, unclear exactly what that means. (sec 4.1, pg 51)

        # if we're using a reference dataframe, (e.g. the full data vs just the
        # training), we want to base our feature sum only off of the full data. If
        # we included this series as well we'd be double counting the training data.

        # NOTE: this is _not_ the l1 norm (I wasn't actually doing l1-norm previously, or at least
        # not along columns - I was doing a row normalization if the sum was > 1, in the
        # first iteration of icat)
        if reference_features is not None:
            feature_sum = reference_features[feature.name].sum()
        else:
            feature_sum = feature.sum()
        if feature_sum == 0 or feature_sum == 0.0:
            return np.zeros(feature.shape)

        # NOTE: this really shouldn't be used - dividing by the sum means that the values are
        # dramatically changed (in general) based on the size of the dataset. This should
        # probably just be a max-norm, kind of like what I'm doing below in _combo_col_row_normalize.

        return feature / feature_sum

    @staticmethod
    def _combo_col_row_normalize(
        features_data: pd.DataFrame, reference_features: pd.DataFrame | None = None
    ):
        """This computes a max-norm-based scaling factor individually for column-wise and row-wise,
        then applying both results. This will always result in row sums less than (but not necessarily
        equal to) 1, and with each feature value scaled relative to the maximum reference value of
        that feature. I don't know if this is a valid normalization method, but it seems to maintain
        the relative scales in both axes while also directly fitting within [0-1] which is necessary
        for anchorviz.
        """
        # NOTE: this function isn't currently in use. _if_ we normalize down below in featurize, we use
        # l1_col_normalize, but the default in model featurize right now is false.

        feature_vals = features_data.vals
        if reference_features is not None:
            reference_vals = reference_features.vals
        else:
            reference_vals = feature_vals

        # NOTE: I guess this col_scale is really only critical for anchors that can be > 1,
        # namely non-binary dictionary anchor in this case.
        col_scale = feature_vals / reference_vals.max(axis=0)
        row_scale = feature_vals / feature_vals.sum(axis=1)

        feature_vals *= col_scale
        feature_vals *= row_scale

        return feature_vals

    def _populate_anchor_types_col(self):
        """This refreshes the "Anchor Types" tab of the anchorlist,
        refreshing the section of currently available anchor types, and
        re-searching for ones that are within scope but haven't been added
        yet.
        """
        children = []
        for anchor_type_dict in self.possible_anchor_types:
            color_picker = v.ColorPicker(
                hide_inputs=True,
                v_model=anchor_type_dict["color"],
                hide_canvas=True,
                hideSliders=True,
                hide_mode_switch=True,
                hide_sliders=True,
                disabled=True,
                show_swatches=True,
                swatches=ANCHOR_COLOR_PALLETE,
                width=250,
            )
            color_picker.on_event(
                "input",
                partial(
                    self._handle_ipv_color_picker_input,
                    anchor_type=anchor_type_dict["ref"],
                ),
            )

            rm_btn = v.Btn(
                small=True,
                icon=True,
                v_on="tooltip.on",
                children=[v.Icon(children=["close"])],
                style_="margin-top: 32px;",
            )
            rm_btn.on_event(
                "click",
                partial(
                    self._handle_ipv_rm_anchor_type_btn_click,
                    type_ref=anchor_type_dict["ref"],
                ),
            )
            rm_btn_tooltip = v.Tooltip(
                bottom=True,
                open_delay=500,
                max_width=400,
                v_slots=[
                    {
                        "name": "activator",
                        "variable": "tooltip",
                        "children": rm_btn,
                    }
                ],
                children=[
                    "Remove this anchor type (will remove all current anchors of this type.)"
                ],
            )

            anchor_type_name = v.TextField(v_model=anchor_type_dict["name"], width=100)
            anchor_type_name.on_event(
                "change",
                partial(
                    self._handle_ipv_anchor_type_name_changed,
                    anchor_type=anchor_type_dict["ref"],
                ),
            )
            anchor_type_name.on_event(
                "blur",
                partial(
                    self._handle_ipv_anchor_type_name_changed,
                    anchor_type=anchor_type_dict["ref"],
                ),
            )

            children.append(
                v.Row(
                    children=[
                        v.Col(
                            children=[
                                anchor_type_name,
                                v.Html(
                                    tag="p",
                                    style_="color: grey; margin-top: -18px; margin-bottom: 2px;",
                                    children=[
                                        v.Html(
                                            tag="small",
                                            children=[f"({anchor_type_dict['ref']})"],
                                        )
                                    ],
                                ),
                                v.Html(
                                    tag="p",
                                    children=[f"{anchor_type_dict['ref'].DESCRIPTION}"],
                                ),
                            ]
                        ),
                        rm_btn_tooltip,
                        v.Spacer(),
                        color_picker,
                    ]
                )
            )

        refresh_btn = v.Btn(
            children=["Refresh unused anchor types"], style_="margin-left: 10px;"
        )
        refresh_btn.on_event("click", lambda w, e, d: self.refresh_anchor_types())
        children.append(v.Row(children=[refresh_btn]))

        for anchor_type in Anchor.anchor_types():
            # check if we've already added it
            found = False
            for anchor_type_dict in self.possible_anchor_types:
                if anchor_type_dict["ref"] == anchor_type:
                    found = True
                    break
            if not found and anchor_type.__qualname__ not in [
                "Anchor",
                "SimilarityAnchorBase",
            ]:
                if anchor_type.NAME != "":
                    default_name = anchor_type.NAME
                else:
                    default_name = anchor_type.__qualname__
                new_anchor_type_name_text = v.TextField(v_model=default_name, width=100)
                add_btn = v.Btn(
                    children=["add"],
                    style_="margin-top: 20px; margin-right: 20px;",
                )
                add_btn.on_event(
                    "click",
                    partial(
                        self._handle_ipv_new_anchor_type_add_click,
                        type_ref=anchor_type,
                        text_field_ref=new_anchor_type_name_text,
                    ),
                )

                children.append(
                    v.Row(
                        children=[
                            v.Col(
                                children=[
                                    new_anchor_type_name_text,
                                    v.Html(
                                        tag="p",
                                        style_="color: grey; margin-top: -18px; margin-bottom: 2px;",
                                        children=[
                                            v.Html(
                                                tag="small",
                                                children=[f"({str(anchor_type)})"],
                                            )
                                        ],
                                    ),
                                    v.Html(
                                        tag="p",
                                        children=[f"{anchor_type.DESCRIPTION}"],
                                    ),
                                ]
                            ),
                            add_btn,
                        ]
                    )
                )

        self.anchor_types_layout.children = children

    def _populate_example_anchor_types_dropdown(self):
        """Update the possible anchor types that can be set as the
        default example anchor type used. (Dropdown in the settings
        tab.)"""
        items = []
        for i, anchor_type_dict in enumerate(self.possible_anchor_types):
            if isinstance(anchor_type_dict["ref"](), SimilarityAnchorBase):
                items.append(dict(text=anchor_type_dict["name"], value=i))
        self.example_anchor_types_dropdown.items = items

    # ============================================================
    # PUBLIC FUNCTIONS
    # ============================================================

    def refresh_anchor_types(self):
        """Re-populate lists of anchor types in the interface."""
        self._populate_anchor_types_col()
        self._populate_example_anchor_types_dropdown()

    def add_anchor_types(self, anchor_types: list[type | dict[str, any]]):
        """Add multiple anchor types to the UI.

        Args:
            anchor_types (list[type|dict[str, any]]): the list of anchor types \
                to add, this can consist of a combination of straight class types \
                and dictionaries containing ``ref`` (the class type), ``name``, \
                and ``color``.

        Example:
            .. code-block:: python

                al = AnchorList(anchor_types=[])
                al.add_anchor_types([
                    icat.anchors.DictionaryAnchor,
                    {"ref": icat.anchors.TFIDFAnchor, "color": "#778899"}
                ])
        """
        for anchor_type in anchor_types:
            if isinstance(anchor_type, type):
                self.add_anchor_type(anchor_type)
            if isinstance(anchor_type, dict):
                ref = anchor_type.pop("ref")
                self.add_anchor_type(ref, **anchor_type)

    def add_anchor_type(
        self, anchor_type: type, name: str = None, color: str = "#777777"
    ):
        """Register the passed anchor type (must be a subclass of ``Anchor``) in
        the UI. This adds a corresponding "add new" button for this type.

        Args:
            anchor_type (type): The class type (of subclass ``Anchor``) to register.
            name (str): The name to associate with the type in the UI.
            color (str): The hex color to use in the CSS for rows in the anchorlist \
                and anchors in anchorviz for this anchor type.
        """
        if name is not None:
            # if the user specified a specific name, use that.
            anchor_type_name = name
        elif anchor_type.NAME != "":
            # otherwise, if there's a static name specified in the class, use _that_.
            anchor_type_name = anchor_type.NAME
        else:
            # otherwise directly use the class name
            anchor_type_name = anchor_type.__qualname__

        new_anchor_type_dict = {
            "name": anchor_type_name,
            "ref": anchor_type,
            "color": color,
        }
        self.possible_anchor_types = [
            *self.possible_anchor_types,
            new_anchor_type_dict,
        ]
        self.refresh_anchor_types()

        # assign anchor type to the default example anchor type if we don't already have one, and
        # this new anchor meets the criteria (subclasses similarity base)
        if "ref" not in self.default_example_anchor_type_dict:
            if isinstance(anchor_type(), SimilarityAnchorBase):
                self.example_anchor_types_dropdown.v_model = (
                    self.possible_anchor_types.index(new_anchor_type_dict)
                )
                self.default_example_anchor_type_dict = new_anchor_type_dict
                self.fire_on_default_example_anchor_type_changed()

    def modify_anchor_type(self, anchor_type: type, key: str, val: str):
        """Change an associated property of the specified anchor type.

        This assumes the passed type has already been registered with ``add_anchor_type``.

        Args:
            anchor_type (type): The anchor type to modify the UI property of.
            key (str): The name of the property to update, use ``color`` or ``name``.
            val (str): The new value to assign to the property.
        """
        prev_anchors = []
        updated_anchor = None
        next_anchors = []

        for anchor_type_dict in self.possible_anchor_types:
            if anchor_type_dict["ref"] == anchor_type:
                anchor_type_dict[key] = val
                updated_anchor = anchor_type_dict
            elif updated_anchor is None:
                prev_anchors.append(anchor_type_dict)
            else:
                next_anchors.append(anchor_type_dict)

        if updated_anchor is None:
            # TODO: error, wasn't found
            pass

        self.possible_anchor_types = [*prev_anchors, updated_anchor, *next_anchors]
        self.fire_on_anchor_types_changed()

        # check if it was the default example anchor and if we need to modify its dictinoary too
        if (
            "ref" in self.default_example_anchor_type_dict
            and self.default_example_anchor_type_dict["ref"] == anchor_type
        ):
            self.default_example_anchor_type_dict[key] = val
            self.fire_on_default_example_anchor_type_changed()

        self.refresh_anchor_types()

    def remove_anchor_type(self, anchor_type: type):
        """Removes this anchor type from the current possible anchor types list, and
        removes any corresponding anchors.

        Args:
            anchor_type (type): The class type of the anchor to exclude from the \
                interface.
        """
        anchors_to_remove = [a for a in self.anchors if type(a) == anchor_type]
        for anchor in anchors_to_remove:
            # NOTE: yes use ==, need strict check, not including inheritance.
            if type(anchor) == anchor_type:
                self.remove_anchor(anchor)

        remaining_anchor_types = []
        for anchor_type_dict in self.possible_anchor_types:
            if anchor_type_dict["ref"] != anchor_type:
                remaining_anchor_types.append(anchor_type_dict)
        self.possible_anchor_types = remaining_anchor_types
        self.fire_on_anchor_types_changed()

        # make sure to unset default exaxmple anchor type if it was this one.
        # TODO: at some point it would be helpful if when this occurs, we automatically
        # search through remaining anchor types and auto re-assign. Not MVP right now.
        if (
            "ref" in self.default_example_anchor_type_dict
            and self.default_example_anchor_type_dict["ref"] == anchor_type
        ):
            self.default_example_anchor_type_dict = {}
            self.fire_on_default_example_anchor_type_changed()

        self.refresh_anchor_types()

    def get_anchor_type_config(self, anchor_type: type):
        """Get the dictionary of UI properties for the specified registered anchor type.

        Returns:
            A dictionary with ``ref``, ``name``, and ``color``, or None if the specified
            type isn't found.
        """
        for anchor_type_dict in self.possible_anchor_types:
            if anchor_type_dict["ref"] == anchor_type:
                return anchor_type_dict
        return None

    def get_unique_anchor_name(self) -> str:
        """Returns a name for a new anchor that won't conflict with any existing."""
        name = "New Anchor"

        # ensure new anchor name is unique
        i = 0
        while name in [anchor.anchor_name for anchor in self.anchors]:
            i += 1
            name = f"New Anchor {i}"

        return name

    @param.depends("anchors", watch=True)
    def refresh_anchors_table(self):
        """Re-populate the list of anchors and coverage stats. This function is
        called automatically anytime the anchors property changes.
        """

        # show or hide the collapse/expand anchors button based on whether there's
        # any anchors or not (otherwise header will cover the button)
        if len(self.anchors) == 0:
            if "visibility" not in self.expand_toggle.style_:
                self.expand_toggle.style_ += "visibility: hidden;"
        else:
            self.expand_toggle.style_ = self.expand_toggle.style_.removesuffix(
                "visibility: hidden;"
            )

        items = []
        for anchor in self.anchors:
            coverage = ""
            pct_negative = ""
            pct_positive = ""
            if anchor.name in self.coverage_info:
                coverage = self.coverage_info[anchor.name]["cov_text"]
                pct_negative = self.coverage_info[anchor.name]["neg_text"]
                pct_positive = self.coverage_info[anchor.name]["pos_text"]

            item = dict(
                color=self.get_anchor_type_config(type(anchor))["color"],
                name=anchor.name,
                anchor_name=anchor._anchor_name_input,
                coverage=coverage,
                pct_negative=pct_negative,
                pct_positive=pct_positive,
                in_viz=anchor._in_view_input,
                in_model=anchor._in_model_input,
                widget=anchor.widget,
                processing=False,
            )
            items.append(item)
        self.table.items = items

    def add_anchor(self, anchor: Anchor):
        """Add the passed anchor to this anchor list.

        Args:
            anchor (Anchor): The anchor to add to the list.

        Important:
            This function should be used rather than directly mutating the internal ``anchors``
            list with ``anchors.append``. Panel/param won't detect this change, and certain
            functionality will probably not work on the frontend.

        Note:
            This triggers the "anchor added" event. You can watch for it by specifying a
            callback function to ``on_anchor_added()``
        """

        # make sure the name is unique
        for other_anchor in self.anchors:
            if other_anchor.anchor_name == anchor.anchor_name:
                anchor.anchor_name = self.get_unique_anchor_name()

        anchor.container = self  # TODO: remove once tfidf anchor is handled
        anchor.on_anchor_changed(self.fire_on_anchor_changed)
        self.anchors.append(anchor)

        self.refresh_anchors_table()

        # make the new anchor expanded in the anchors table
        self.table.expanded = [
            *self.table.expanded,
            dict(name=self.table.items[-1]["name"]),
        ]

        self.fire_on_anchor_added(anchor)

    def remove_anchor(self, anchor: Anchor):
        """Remove the specified anchor from this anchor list.

        Args:
            anchor (Anchor): The anchor to remove from the list.

        Important:
            This function should be used rather than directly mutating the internal
            ``anchors`` list with ``anchors.remove``. Panel/param won't detect this
            change, and certain functionality will probably not work on the frontend.

        Note:
            This triggers the "anchor removed" event. You can watch for it by specifying
            a callback function to ``on_anchor_removed()``
        """
        self.anchors.remove(anchor)
        self.refresh_anchors_table()
        self.fire_on_anchor_removed(anchor)

    def set_coverage(self, coverage_info: dict[str, dict[str, int | float]]):
        """Set the anchor coverage data, to be updated and displayed in the table.

        Args:
            coverage_info (dict[str, dict[str, Union[int, float]]]): Dictionary (keys \
                being the anchor panel ids) and the value being a dictionary with the \
                "row" of data to display in the table.  Keys expected: "total", "pos", \
                "neg", "total_pct", "pos_pct", "neg_pct"
        """
        self.coverage_info = coverage_info

        for anchor_name in self.coverage_info:
            cov_pct_text = str(float(coverage_info[anchor_name]["total_pct"]) * 100.0)[
                0:4
            ]
            if self.model.is_seeded():
                pos_pct_text = str(
                    float(coverage_info[anchor_name]["pos_pct"]) * 100.0
                )[0:4]
                neg_pct_text = str(
                    float(coverage_info[anchor_name]["neg_pct"]) * 100.0
                )[0:4]

            cov_txt = f"{cov_pct_text}% ({int(coverage_info[anchor_name]['total'])})"
            pos_txt = ""
            neg_txt = ""

            if self.model.is_seeded():
                pos_txt = f"{pos_pct_text}% ({int(coverage_info[anchor_name]['pos'])})"
                neg_txt = f"{neg_pct_text}% ({int(coverage_info[anchor_name]['neg'])})"

            self.coverage_info[anchor_name]["cov_text"] = cov_txt
            self.coverage_info[anchor_name]["pos_text"] = pos_txt
            self.coverage_info[anchor_name]["neg_text"] = neg_txt

        self.refresh_anchors_table()

    def get_anchor_by_panel_id(self, panel_id: str) -> Anchor:
        """Get the anchor instance with the associated panel name/id.

        Args:
            panel_id (str): The panel ID string of the anchor, the format \
                should be DictinaryAnchor00XX or similar.
        """
        for anchor in self.anchors:
            if anchor.name == panel_id:
                return anchor
        return None

    def featurize(
        self,
        data: pd.DataFrame,
        normalize: bool = True,
        reference_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Apply the featurization from each anchor to the passed data.

        Args:
            data (pd.DataFrame): The data to compute the features on.
            normalize (bool): Whether to apply L1 norm (column-wise) to the features.
            reference_data (Optional[pd.DataFrame]): If normalizing, use this data to determine feature sums.

        Returns:
            The featured dataframe.
        """
        self.table.processing = True
        features = []
        for anchor in self.anchors:
            self.table._set_anchor_processing(anchor.name, True)
            data[f"_{anchor.anchor_name}"] = anchor.featurize(data) * anchor.weight
            features.append(f"_{anchor.anchor_name}")
            self.table._set_anchor_processing(anchor.name, False)
        if normalize:
            if reference_data is not None:
                data.loc[:, features] = data[features].apply(
                    self._l1_col_normalize,
                    axis=0,
                    reference_features=reference_data[features],
                )
            else:
                data.loc[:, features] = data[features].apply(
                    self._l1_col_normalize, axis=0
                )
        self.table.processing = False

        return data

    def highlight_regex(self) -> str:
        """Construct a regex for all keywords in dictionary anchors, for
        use in highlighting keywords in a text.

        Returns:
            A regex string that is essentially just the "|" or'd regexes
            of the individual anchors.
        """

        kw_regex = "|".join(
            [
                anchor.regex()
                for anchor in self.anchors
                if type(anchor) == DictionaryAnchor and anchor.regex() != ""
            ]
        )
        kw_regex = f"({kw_regex})" if kw_regex != "" else kw_regex
        return kw_regex

    def build_tfidf_features(self):
        # TODO: this eventually needs to use the anchorlist cache, and also
        # this may need to change when tfidf just becomes a sim function instead
        # of a dedicated anchor
        if self.model is None:
            raise RuntimeError(
                "The anchorlist has no associated model to get a dataset from."
            )
        if self.model.data.active_data is None:
            return

        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        # TODO: coupling: accessing active_data through model
        self.tfidf_features = self.tfidf_vectorizer.fit_transform(
            self.model.data.active_data[self.model.data.text_col]
        )

    def save(self, path: str):
        """Save the configuration for each individual anchor, and pickle the cache,
        at the specified location.
        """
        anchors_info = []

        os.makedirs(f"{path}/anchors", exist_ok=True)

        # save the pickled list of anchor types
        with open(f"{path}/anchorlist_anchortypes.pkl", "wb") as outfile:
            pickle.dump(self.possible_anchor_types, outfile)

        # save the current settings
        with open(f"{path}/anchorlist_settings.json", "w") as outfile:
            settings = dict(
                example_anchor_type_index=self.example_anchor_types_dropdown.v_model
            )
            json.dump(settings, outfile)

        # save each individual anchor and collect its information
        for anchor in self.anchors:
            name = anchor.anchor_name
            anchor_info = {
                "module": anchor.__module__,
                "classname": anchor.__class__.__qualname__,
                "params_path": f"{path}/anchors/{name}",
                "theta": 0.0,
            }

            # get theta from the anchorviz anchor list
            if self.model is not None:
                for av_anchor_info in self.model.view.anchorviz.anchors:
                    if av_anchor_info["id"] == anchor.name:
                        anchor_info["theta"] = av_anchor_info["theta"]
                        break
            anchor.save(f"{path}/anchors/{name}")
            anchors_info.append(anchor_info)

        # save the cache
        with open(f"{path}/anchorlist_cache.pkl", "wb") as outfile:
            pickle.dump(self.cache, outfile)

        # save the anchor information
        with open(f"{path}/anchorlist_anchors.json", "w") as outfile:
            json.dump(anchors_info, outfile, indent=4)

    def load(self, path: str):
        """Reload parameters for and re-add all anchors from specified location, as
        well as unpickle any previously saved cache.
        """

        # clear any existing anchor types first
        for anchor_type_dict in self.possible_anchor_types:
            self.remove_anchor_type(anchor_type_dict["ref"])

        # load anchor types
        with open(f"{path}/anchorlist_anchortypes.pkl", "rb") as infile:
            anchor_types = pickle.load(infile)
            self.add_anchor_types(anchor_types)

        # load anchor settings
        with open(f"{path}/anchorlist_settings.json") as infile:
            settings = json.load(infile)
            self.example_anchor_types_dropdown.v_model = settings[
                "example_anchor_type_index"
            ]
            self.default_example_anchor_type_dict = self.possible_anchor_types[
                settings["example_anchor_type_index"]
            ]
            self.fire_on_default_example_anchor_type_changed()

        # load the anchor information
        with open(f"{path}/anchorlist_anchors.json") as infile:
            anchors_info = json.load(infile)

        # load the cache
        with open(f"{path}/anchorlist_cache.pkl", "rb") as infile:
            self.cache = pickle.load(infile)

        # load and re-add each individual anchor
        for anchor_info in anchors_info:
            # black magic import to get the class type (this is to make it
            # possible for user to implement their own anchor classes and still
            # be handled correctly here)
            module = __import__(anchor_info["module"])
            klass = getattr(module, anchor_info["classname"])

            # initialize and load the anchor type
            anchor = klass()
            anchor.load(anchor_info["params_path"])
            anchor.theta = anchor_info["theta"]
            self.add_anchor(anchor)
