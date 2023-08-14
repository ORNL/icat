"""The AnchorList component class file, these manage a model's current set of anchors."""

import json
import os
import pickle
from collections.abc import Callable

import ipyvuetify as v
import ipywidgets as ipw
import numpy as np
import pandas as pd
import panel as pn
import param
import traitlets
from sklearn.feature_extraction.text import TfidfVectorizer

from icat.anchors import Anchor, DictionaryAnchor, SimilarityFunctionAnchor, TFIDFAnchor
from icat.utils import _kill_param_auto_docstring

_kill_param_auto_docstring()


class AnchorListTemplate(v.VuetifyTemplate):
    """The ipyvuetify contents of the anchorlist table. This handles all the special considerations
    like the slot for the expanded rows containing the anchor widgets, and the coverage v-html etc.
    """

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

    # TODO: set_all_expanded, set_all_collapsed

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

    @traitlets.default("template")
    def _template(self):
        return """
        <v-data-table
            :headers="headers"
            :items="items"
            hide-default-footer
            show-expand
            dense
            :loading="processing"
            :expanded.sync="expanded"
            class="dense-table striped-table softhover-table"
            item-key="name"
        >
            <template v-slot:item.anchor_name="{ item }">
                <div><jupyter-widget :widget="item.anchor_name" /></div>
            </template>

            <template v-slot:item.in_viz="{ item }">
                <div><jupyter-widget :widget="item.in_viz" /></div>
            </template>

            <template v-slot:item.in_model="{ item }">
                <div><jupyter-widget :widget="item.in_model" /></div>
            </template>

            <template v-slot:item.pct_negative="{ item }">
                <div class='orange--text darken-1'>{{ item.pct_negative }}</div>
            </template>

            <template v-slot:item.pct_positive="{ item }">
                <div class='blue--text darken-1'>{{ item.pct_positive }}</div>
            </template>

            <template v-slot:item.delete="{ item }">
                <v-btn
                    icon
                    x-small
                    class='delete-button'
                    @click="deleteAnchor(item.name)"
                    :loading="item.processing"
                >
                    <v-icon>mdi-close-circle-outline</v-icon>
                </v-btn>
            </template>

            <template v-slot:expanded-item="{ headers, item }">
                <td :colspan="headers.length">
                    <jupyter-widget :widget="item.widget" />
                </td>
            </template>
        </v-data-table>
        <style id='table-styles'>
            .softhover-table table tbody tr:hover {
                background-color: #333333 !important;
            }
            .delete-button {
                margin: 0px;
                margin-left: 6px;
                color: var(--md-grey-500) !important;
            }
            .delete-button:hover {
                color: var(--md-red-500) !important;
            }
            .striped-table tbody tr:nth-child(even) {
                background-color: rgba(0, 0, 0, 0.35);
            }
            .striped-table .v-data-table__expanded__content td {
                background-color: #263238;
            }
            .dense-table .row {
                flex-wrap: nowrap;
            }
            .dense-table td {
                padding: 0 4px !important;
                height: 30px !important;
                max-height: 30px !important;
                vertical-align: middle;
            }
            .dense-table th {
                padding: 0 4px !important;
            }
            .dense-table td .v-input {
                margin: 0;
                /* margin-top: 5px; */
            }
            .dense-table td .v-input__slot {
                margin-bottom: 0;
            }
            .dense-table td .v-input--selection-controls__input {
                margin-top: -5px;
            }
            .dense-table td .v-input--selection-controls__ripple {
                margin: 7px;
                height: 25px !important;
                width: 25px !important;
            }
            .dense-table td .v-icon.v-icon::after {
                transform: scale(1.2) !important;
            }
            .dense-table td .v-input .v-messages {
                display: none;
                height: 0;
            }
            .dense-table td .v-text-field__details {
                height: 2px !important;
                min-height: 2px !important;
            }
            div .v-progress-linear {
                left: -1px !important;
            }
        </style>
        """


class AnchorList(pn.viewable.Layoutable, pn.viewable.Viewer):
    """A model's list tracking and managing a collection of anchors.

    This is what handles creating features for a dataset. This class is also a
    visual component for interacting with and modifying those anchors in a table
    format, and is used as part of the greater model interactive view.

    Args:
        model: The parent model.
        table_width (int): Static width of the visual component table.
        table_height (int): Static height of the visual component table.
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

    # TODO: coupling: model is used to retrieve model.data.active_data
    # NOTE: one way around this would be to have a "reference_data" property,
    # where the setter (which could be hooked up through a model event) re-
    # computes relevant tfidf vals
    def __init__(
        self,
        model=None,
        table_width: int = 700,
        table_height: int = 150,
        **params,
    ):
        super().__init__(**params)  # required for panel components

        self.coverage_info = {}
        """Dictionary associating panel id of anchor with dictionary of 'coverage', 'pct_positive', and 'pct_negative'"""

        self.expand_toggle = v.Btn(
            icon=True,
            fab=True,
            x_small=True,
            plain=True,
            children=[v.Icon(children=["mdi-expand-all"])],
        )
        self.expand_toggle.on_event("click", self._handle_ipv_expand_toggle_click)

        # creating objects for adding new anchors to list
        self.dictionary_button = pn.widgets.Button(
            name="Add Dictionary Anchor",
            button_type="primary",
        )
        self.dictionary_button.on_click(self._handle_pnl_new_dictionary_btn_clicked)
        self.tfidf_button = pn.widgets.Button(
            name="Add TF-IDF Anchor",
            button_type="primary",
        )
        self.tfidf_button.on_click(self._handle_pnl_new_tfidf_btn_clicked)

        self.similarity_button = pn.widgets.Button(
            name="Add Similarity Anchor",
            button_type="primary",
        )
        self.similarity_button.on_click(self._handle_pnl_new_similarity_btn_clicked)

        self.table = AnchorListTemplate()
        self.table.on_anchor_removal(self._handle_table_anchor_deleted)

        self.layout = pn.Column(
            pn.Row(
                self.expand_toggle,
                self.dictionary_button,
                self.tfidf_button,
                self.similarity_button,
            ),
            self.table,
            height=table_height,
            width=table_width,
        )
        """The full component layout for panel to display."""

        self.coverage: dict[dict] = None
        """Keys are anchor ids, value is a dictionary of metrics to display"""

        self.cache: dict[str, any] = {}
        """This cache gets pickled on save, useful for anchors to store results of
        processing-intensive tasks to reduce featurizing time."""

        self.tfidf_vectorizer = None
        self.tfidf_features = None

        self.model = model

        # callback collections
        self._anchor_changed_callbacks: list[Callable] = []
        self._anchor_added_callbacks: list[Callable] = []
        self._anchor_removed_callbacks: list[Callable] = []

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def _handle_table_anchor_deleted(self, name: str):
        """Event handler for the list table template delete button"""
        anchor = self.get_anchor_by_panel_id(name)
        self.remove_anchor(anchor)

    def _handle_pnl_new_dictionary_btn_clicked(self, event):
        # TODO: since the renaming logic will be basically the same, we should probably
        # move that to the add_anchor logic instead.
        name = self.get_unique_anchor_name()
        self.add_anchor(DictionaryAnchor(anchor_name=name))

    def _handle_pnl_new_tfidf_btn_clicked(self, event):
        name = self.get_unique_anchor_name()
        self.add_anchor(TFIDFAnchor(anchor_name=name))

    def _handle_pnl_new_similarity_btn_clicked(self, event):
        name = self.get_unique_anchor_name()
        self.add_anchor(SimilarityFunctionAnchor(anchor_name=name))

    def _handle_ipv_expand_toggle_click(self, widget, event, data):
        # if any are expanded, set expanded to blank list
        if len(self.table.expanded) > 0:
            self.table.expanded = []
        else:
            self.table.expanded = [{"name": item["name"]} for item in self.table.items]

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

    def fire_on_anchor_added(self, anchor: Anchor):
        for callback in self._anchor_added_callbacks:
            callback(anchor)

    def fire_on_anchor_removed(self, anchor: Anchor):
        for callback in self._anchor_removed_callbacks:
            callback(anchor)

    def fire_on_anchor_changed(self, name: str, key: str, value):
        """Trigger the event to notify that a property on an anchor changed.

        Note that this is usually used for simply passing along the change directly
        from the anchor's individual on_anchor_changed events

        Args:
            name (str): the *internal name* that panel is using, which we're using as
            the anchor id.
            key (str): the name of the property being changed.
            value (any): The value the property was changed to.
        """
        for callback in self._anchor_changed_callbacks:
            callback(name, key, value)

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
        for anchorviz."""

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

    # ============================================================
    # PUBLIC FUNCTIONS
    # ============================================================

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
        """Re-populate the list of anchors and coverage stats."""
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
            This function should be used rather than directly mutating the internal :code:`anchors`
            list with :code:`anchors.append`. Panel/param won't detect this change, and certain functionality
            will probably not work on the frontend.

        Note:
            This triggers the "anchor added" event. You can watch for it by specifying a callback function
            to :code:`on_anchor_added()`
        """
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
            This function should be used rather than directly mutating the internal :code:`anchors`
            list with :code:`anchors.remove`. Panel/param won't detect this change, and certain functionality
            will probably not work on the frontend.

        Note:
            This triggers the "anchor removed" event. You can watch for it by specifying a callback function
            to :code:`on_anchor_removed()`
        """
        self.anchors.remove(anchor)
        self.refresh_anchors_table()
        self.fire_on_anchor_removed(anchor)

    def set_coverage(self, coverage_info: dict[str, dict[str, int | float]]):
        """Set the anchor coverage data, to be updated and displayed in the table.

        Args:
            coverage_info (dict[str, dict[str, Union[int, float]]]): Dictionary (keys being the
                anchor panel ids) and the value being a dictionary with the "row" of data to
                display in the table.
                Keys expected: "total", "pos", "neg", "total_pct", "pos_pct", "neg_pct"
        """
        self.coverage_info = coverage_info

        for anchor_name in self.coverage_info:
            cov_pct_text = str(float(coverage_info[anchor_name]["total_pct"]) * 100.0)[
                0:4
            ]
            pos_pct_text = str(float(coverage_info[anchor_name]["pos_pct"]) * 100.0)[
                0:4
            ]
            neg_pct_text = str(float(coverage_info[anchor_name]["neg_pct"]) * 100.0)[
                0:4
            ]

            cov_txt = f"{cov_pct_text}% ({int(coverage_info[anchor_name]['total'])})"
            pos_txt = f"{pos_pct_text}% ({int(coverage_info[anchor_name]['pos'])})"
            neg_txt = f"{neg_pct_text}% ({int(coverage_info[anchor_name]['neg'])})"

            self.coverage_info[anchor_name]["cov_text"] = cov_txt
            self.coverage_info[anchor_name]["pos_text"] = pos_txt
            self.coverage_info[anchor_name]["neg_text"] = neg_txt

        self.refresh_anchors_table()

    def get_anchor_by_panel_id(self, panel_id: str) -> Anchor:
        """Get the anchor instance with the associated panel name/id."""
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
        use in highlighting keywords in a text."""

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
        at the specified location."""
        anchors_info = []

        os.makedirs(f"{path}/anchors", exist_ok=True)

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
        with open(f"{path}/anchorlist.json", "w") as outfile:
            json.dump(anchors_info, outfile, indent=4)

    def load(self, path: str):
        """Reload parameters for and re-add all anchors from specified location, as
        well as unpickle any previously saved cache."""
        # load the anchor information
        with open(f"{path}/anchorlist.json") as infile:
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
