"""The parent anchor class and a few anchor types.

Anchors are the interactive featuring component of ICAT, an anchor class
determines how a feature is computed on the data, and the user interacts with
the UI widget of the anchor to modify the feature (e.g. what keywords it
searches for.)
"""

# NOTE: "container" refers to the containing anchorlist instance

import json
from collections.abc import Callable

import ipyvuetify as v
import numpy as np
import pandas as pd
import panel as pn
import param
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import icat.utils

icat.utils._kill_param_auto_docstring()


# TODO: coupling: we don't need access to container, just have a "fire_anchor_changed"
# and "on_anchor_changed" that the container listens to.
class Anchor(param.Parameterized):
    """The main parent anchor class, this should be subclassed, not directly used.

    Args:
        container (AnchorList): The anchor list that this anchor is a part of. If
            you are creating the anchor manually, leave this None, it will get
            populated automatically.
        anchor_name (str): The label to show for this anchor.
        weight (float): Scalar multiple to apply to the output feature, this modifies
            how strongly a particular feature is likely to influence the model.
        in_view (bool): Whether to show this anchor in anchorviz.
        in_model (bool): Whether to include this feature in the training process.
    """

    anchor_name = param.String(default="New Anchor")
    """Not to be confused with just ``name``, which is the panel component id."""
    weight = param.Number(1.0, bounds=(0.0, 5.0))
    """Scalar multiple to apply to all output features, the user can change this
    to modify how much a particular feature influences the model."""

    in_view = param.Boolean(True, precedence=-1)
    """Whether to show this anchor in anchorviz."""
    in_model = param.Boolean(True, precedence=-1)
    """Whether to include this feature in the training process. (it will be included
    in the featurization call and will update locations inside anchorviz, it simply
    won't be passed to the model.)"""

    def __init__(self, container=None, **params):
        self.container = container
        # self.processing = True
        # self.event_to_trigger = None
        super().__init__(**params)

        self._anchor_name_input = v.TextField(
            dense=True, label="Anchor Name", v_model=self.anchor_name, single_line=True
        )
        # NOTE: "input" fires for every keypress. We use the input event for this one
        # because the model specially handles an anchor name change and _doesn't_ refit
        # the model for that change. Most other text field events will want to fire on
        # both "blur" (lost focus, e.g. clicked away, and we assume they're done) and
        # "change".
        self._anchor_name_input.on_event("input", self._handle_ipv_anchor_name_changed)
        self._anchor_name_input.on_event("blur", self._handle_ipv_anchor_name_changed)

        self._weight_input = v.Slider(
            label="Weight",
            min="0",
            max="5",
            thumb_label=True,
            step="0.1",
            v_model=self.weight,
            dense=True,
        )
        self._weight_input.on_event("change", self._handle_ipv_weight_input_changed)

        self._in_view_input = v.Checkbox(v_model=self.in_view, dense=True)
        self._in_view_input.on_event("change", self._handle_ipv_in_view_input_changed)
        self._in_model_input = v.Checkbox(v_model=self.in_model, dense=True)
        self._in_model_input.on_event("change", self._handle_ipv_in_model_input_changed)

        self._anchor_changed_callbacks: list[Callable] = []

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def _handle_ipv_anchor_name_changed(self, widget, event, data):
        """Ipyvuetify event handler for the anchor name text field."""
        self.anchor_name = self._anchor_name_input.v_model

    def _handle_ipv_weight_input_changed(self, widget, event, data):
        """Ipyvuetify event handler. Note that this is one direction, the regular param
        event handler has to handle going the other direction (setting weight input
        based on new weight)"""
        self.weight = self._weight_input.v_model

    def _handle_ipv_in_view_input_changed(self, widget, event, data):
        """Ipyvuetify event handler for in_view checkbox."""
        self.in_view = self._in_view_input.v_model

    def _handle_ipv_in_model_input_changed(self, widget, event, data):
        """Ipyvuetify event handler for in_model checkbox."""
        self.in_model = self._in_model_input.v_model

    @param.depends("anchor_name", watch=True)
    def _handle_pnl_name_changed(self):
        self._anchor_name_input.v_model = self.anchor_name
        self.fire_on_anchor_changed("anchor_name", self.anchor_name)

    @param.depends("weight", watch=True)
    def _handle_pnl_weight_changed(self):
        self._weight_input.v_model = self.weight
        self.fire_on_anchor_changed("weight", self.weight)

    @param.depends("in_view", watch=True)
    def _handle_pnl_in_view_changed(self):
        self._in_view_input.v_model = self.in_view
        self.fire_on_anchor_changed("in_view", self.in_view)

    @param.depends("in_model", watch=True)
    def _handle_pnl_in_model_changed(self):
        self._in_model_input.v_model = self.in_model
        self.fire_on_anchor_changed("in_model", self.in_model)

    # ============================================================
    # EVENT SPAWNERS
    # ============================================================

    def on_anchor_changed(self, callback: Callable):
        """Register a callback for the anchor_changed event.

        Callbacks for this event should take three parameters:
        * Name (string) (this is the internal panel name, which we use as the anchor id.)
        * Property name (string) that's changing on the anchor.
        * Value that the property on the anchor was changed to.
        """
        self._anchor_changed_callbacks.append(callback)

    def fire_on_anchor_changed(self, key: str, value):
        for callback in self._anchor_changed_callbacks:
            callback(self.name, key, value)

    # ============================================================
    # PUBLIC FUNCTIONS
    # ============================================================

    def featurize(self, data: pd.DataFrame) -> pd.Series:
        """Create an anchor weight from the passed dataframe.

        Note:
            Expected to be overridden in subclasses.
        """
        raise NotImplementedError()

    def row_view(self) -> pn.Row:
        return pn.Row(self.param)

    def to_dict(self) -> dict[str, any]:
        """Get a dictionary of this anchor's parameters. Useful for easily implementing
        save functionality."""
        return dict(
            anchor_name=self.anchor_name,
            weight=self.weight,
            in_view=self.in_view,
            in_model=self.in_model,
        )

    # def save(self, path: str):
    #     """Save the information for this anchor at the specified path, such that calling
    #     load() with the same path will return all values as they were.

    #     Note:
    #         Expected to be overriden in subclasses. It's recommended that you implement
    #         a ``to_dict`` function that updates a dictionary from the parent anchor's
    #         ``to_dict`` function to get all relevant information.
    #     """
    #     raise NotImplementedError()

    # def load(self, path: str):
    #     """Load an anchor's parameters from the specified path.

    #     Note:
    #         Expected to be overriden in subclasses.
    #     """
    #     raise NotImplementedError()

    def save(self, path: str):
        """Save anchor to specified path."""
        with open(f"{path}.json", "w") as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

    def load(self, path: str):
        """Load anchor from specified path."""
        with open(f"{path}.json") as infile:
            params = json.load(infile)
        icat.utils.populate_anchor_from_dictionary(self, params)


class DictionaryAnchor(Anchor):
    """A bag-of-words feature that returns raw count value sum of the
    number of occurrences of each word in the given keywords."""

    keywords_str = param.String(label="Keywords")
    text_col = param.String(precedence=-1)

    keywords = param.List(precedence=-1)

    def __init__(self, container=None, **params):
        if "keywords" in params:
            self.keywords_str = ",".join(params["keywords"])
        super().__init__(container, **params)

        self._keywords_input = v.TextField(
            label="Keywords", v_model=self.keywords_str, dense=True, single_line=True
        )
        self._keywords_input.on_event("change", self._handle_ipv_keywords_input_change)
        self._keywords_input.on_event("blur", self._handle_ipv_keywords_input_change)

        self.widget = v.Container(
            fluid=True,
            dense=True,
            children=[
                v.Row(
                    dense=True,
                    children=[
                        v.Col(
                            dense=True,
                            style_="padding: 0;",
                            class_="col-4",
                            children=[self._weight_input],
                        ),
                        v.Col(
                            dense=True,
                            style_="padding: 0;",
                            children=[self._keywords_input],
                        ),
                    ],
                )
            ],
        )

    def _handle_ipv_keywords_input_change(self, widget, event, data):
        """Ipyvuetify event handler, triggers param event handler."""
        self.keywords_str = self._keywords_input.v_model

    @param.depends("keywords_str", watch=True)
    def _handle_pnl_keywords_str_change(self):
        """We propagate the change to the keywords"""
        self.keywords = self.keywords_str.split(",")

    @param.depends("keywords", watch=True)
    def _handle_pnl_keywords_to_keywords_str(self):
        """This could either be directly from a keywords change, or from a keywords_str change.
        We enforce consistency with the latter but do not trigger the keywords_str event (to prevent
        infinite looping), and trigger the anchorlist event."""
        # TODO: docstring wrong, param is smarter than me.
        # with param.parameterized.discard_events(self):
        self.keywords_str = ",".join(self.keywords)
        self._keywords_input.v_model = self.keywords_str

        self.fire_on_anchor_changed("keywords", self.keywords)

    def _keyword_count(self, text: str) -> float:
        total = 0

        for word in self.keywords:
            if word.strip() != "":
                total += text.lower().count(word.lower())
        return total * self.weight

    def featurize(self, data: pd.DataFrame) -> pd.Series:
        return data[self.text_col].apply(self._keyword_count)

    def _sanitize_regex_symbols(self, string: str) -> str:
        # string = string.replace('\', '\\')
        string = string.replace(r"*", r"\*")
        string = string.replace(r"(", r"\(")
        string = string.replace(r")", r"\)")
        string = string.replace(r"|", r"\|")
        string = string.replace(r"[", r"\[")
        string = string.replace(r"]", r"\]")
        string = string.replace("-", r"\-")
        string = string.replace("+", r"\+")
        return string

    def regex(self) -> str:
        """Return a regex string that would capture what this anchor is featurizing on."""
        return "|".join(
            [
                self._sanitize_regex_symbols(keyword)
                for keyword in self.keywords
                if keyword != ""
            ]
        )

    def row_view(self) -> pn.Row:
        return pn.Row(
            self.param.weight,
            self.param.keywords_str,
        )

    def to_dict(self) -> dict[str, any]:
        """Get a dictionary of all relevant parameters that define this anchor."""
        params = super().to_dict()
        self_params = dict(
            keywords_str=self.keywords_str,
            text_col=self.text_col,
            keywords=self.keywords,
        )
        params.update(self_params)
        return params


class SimilarityAnchorBase(Anchor):
    # TODO: have to have the texts _and_ the indices from the original data
    # that this is based on? (because the indices are what we'll want to base
    # the widget off of, but the texts should be what we actually care about/store)
    # is there concern about these getting off from each other?

    # alternatively, maybe just do away with indices entirely? The goal for
    # anchors is to be relatively pure

    text_col = param.String(precedence=-1)

    reference_texts = param.List([])

    reference_short = param.List([])
    """A shortform version of the texts, either the row IDs if available, or
    just the first few words"""

    def __init__(self, container=None, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self._chips_container = v.Container(
            dense=True, children=[], style_="padding: 0;"
        )  # munch munch

        self._id_text = v.TextField(
            label="Row ID", v_model="", dense=True, single_line=True, width=40
        )
        self._add_button = v.Btn(children=["Add"])
        self._add_button.on_event("click", self._handle_ipv_add_btn_clicked)
        self._id_text.on_event("change", self._handle_ipv_id_text_changed)

        self.new_id = 0

        self.widget = v.Container(
            fluid=True,
            dense=True,
            children=[
                v.Row(
                    dense=True,
                    children=[
                        v.Col(
                            dense=True,
                            style_="padding: 0;",
                            class_="col-4",
                            children=[self._weight_input],
                        ),
                        v.Col(
                            dense=True,
                            style_="padding: 0;",
                            children=[self._chips_container],
                        ),
                        v.Col(
                            dense=True,
                            style_="padding: 0",
                            class_="col-2",
                            children=[self._id_text],
                        ),
                        v.Col(
                            dense=True,
                            style_="padding: 0",
                            class_="col-2",
                            children=[self._add_button],
                        ),
                    ],
                )
            ],
        )

    def _handle_ipv_id_text_changed(self, widget, event, data):
        self._add_id_text_to_reference_texts()

    def _handle_ipv_add_btn_clicked(self, widget, event, data):
        self._add_id_text_to_reference_texts()

    def _add_id_text_to_reference_texts(self):
        """Takes whatever was put in the id text and adds either the raw text, or
        the text at the specified ID (if it was an actual id given and we have
        data.)"""
        self.new_id = self._id_text.v_model

        # get the text of the row id specified, or use as the text itself if not an id/no model
        if (
            self.container is not None
            and self.container.model is not None
            and self.new_id.isdigit()
        ):
            active_data = self.container.model.data.active_data
            new_text = active_data.loc[int(self.new_id), self.text_col]
        else:
            new_text = self.new_id
        self._id_text.v_model = ""
        self.reference_texts = [*self.reference_texts, new_text]

    @param.depends("reference_texts", watch=True)
    def _handle_pnl_texts_change(self):
        # get texts from container?

        # get the data manager if it exists (so we can convert to ids)
        data = None
        if self.container is not None and self.container.model is not None:
            data = self.container.model.data.active_data

        refs = []

        for text in self.reference_texts:
            if data is not None:
                # first check the data for an exact text match and convert to id
                search = data[data[self.text_col] == text]
                if search.shape[0] > 0:
                    refs.append(str(search.iloc[0].name))
                else:
                    # otherwise treat it as a raw text
                    refs.append(text[:25])
            else:
                # treat as raw text if no active dataset
                refs.append(text[:25])

        self.reference_short = refs

    @param.depends("reference_short", watch=True)
    def _handle_pnl_shorts_change(self):
        chips = []
        for index, short in enumerate(self.reference_short):
            chip = v.Chip(close_=True, children=[short], v_on="tooltip.on")
            chip.on_event("click:close", self._handle_ipv_chip_close)
            # TODO: when you click on it, it should put it in the instance
            # chip.on_event("")

            tooltip = v.Tooltip(
                top=True,
                v_slots=[
                    {"name": "activator", "variable": "tooltip", "children": chip}
                ],
                max_width=400,
                children=[self.reference_texts[index]],
            )

            chips.append(tooltip)
        self._chips_container.children = chips

        self.fire_on_anchor_changed("reference_short", self.reference_short)

    def _handle_ipv_chip_close(self, widget, event, data):
        self.remove_by_short(widget.children[0])

    def remove_by_short(self, short_text):
        """Delete one of the reference texts based on the short version."""
        index = self.reference_short.index(short_text)
        if index != -1:
            new_list = []
            for i, item in enumerate(self.reference_texts):
                if i != index:
                    new_list.append(item)
            self.reference_texts = new_list

    def featurize(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def to_dict(self) -> dict[str, any]:
        """Get a dictionary of all relevant parameters that define this anchor."""
        params = super().to_dict()
        self_params = dict(
            text_col=self.text_col,
            reference_texts=self.reference_texts,
            reference_short=self.reference_short,
        )
        params.update(self_params)
        return params


class TFIDFAnchor(SimilarityAnchorBase):
    def __init__(self, container=None, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

    def _tfidf_similarities(self, vector, comparison_vectors):
        # TODO: what is instance/instances?
        # these are already numpy arrays/the tfidf vectors I think.
        sims = cosine_similarity(vector, comparison_vectors)
        return sims

    # TODO: I really need more integration tests for this, there's a lot
    # of conditions going on here.
    def featurize(self, data: pd.DataFrame) -> pd.Series:
        if len(self.reference_texts) == 0:
            return pd.Series(0, index=data.index)

        # check to see if the containing anchorlist has a reference TfidfVectorizer/
        # tfidf_features set
        tfidf_vectorizer = None
        tfidf_features = None
        reference_features: np.ndarray = None
        if self.container is not None:
            if self.container.tfidf_vectorizer is not None:
                tfidf_vectorizer = self.container.tfidf_vectorizer

                if len(data) == self.container.tfidf_features.shape[0]:
                    tfidf_features = self.container.tfidf_features

        # if it doesn't, create a new one and fit/transform it on the data
        if tfidf_vectorizer is None:
            tfidf_vectorizer = TfidfVectorizer(stop_words="english")

            if tfidf_features is None:
                # This only triggers if we don't even have a vectorizer on the container?
                tfidf_features = tfidf_vectorizer.fit_transform(data[self.text_col])
        elif tfidf_features is None:
            # if we have a vectorizer but the data is different then what was
            # initially used, we only want to transform, because a fit_transform
            # will potentially create very differently shaped vectors
            tfidf_features = tfidf_vectorizer.transform(data[self.text_col])

        reference_features = tfidf_vectorizer.transform(self.reference_texts)

        combined_reference_features = reference_features.mean(axis=0)
        similarities = self._tfidf_similarities(
            np.asarray(combined_reference_features), tfidf_features
        )

        # store/save/cache the vectorizer and features
        # NOTE: we don't really have a way to detect when the data has sufficiently
        # changed that we should refit/transform vectorizer and features.
        # if self.container is not None:
        #     self.container.tfidf_vectorizer = tfidf_vectorizer
        #     self.container.tfidf_features = tfidf_features

        return pd.Series(similarities[0], index=data.index)


# TODO: probably instead of this extending TF-IDFAnchor, we should have them extend from same root?
class SimilarityFunctionAnchor(SimilarityAnchorBase):
    similarity_function = param.String("")

    # TODO: need a dropdown for the possible similarity functions
    def __init__(self, container=None, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self.sim_function_options = v.Select(label="Similarity function", items=[{}])
        self.sim_function_options.on_event(
            "change", self._handle_ipv_sim_function_change
        )

        self.widget.children = [
            *self.widget.children,
            v.Row(
                dense=True,
                children=[self.sim_function_options],
            ),
        ]
        self._populate_items()

    # TODO: need dropdown event handlers to modify similarity_function

    def _handle_ipv_sim_function_change(self, widget, event, data):
        self.similarity_function = data
        self.fire_on_anchor_changed("similarity_function", data)

    def _populate_items(self):
        items = []
        if self.container is not None and self.container.model is not None:
            items = list(self.container.model.similarity_functions.keys())
        self.sim_function_options.items = items

    def featurize(self, data: pd.DataFrame) -> pd.Series:
        if len(self.reference_texts) == 0:
            return pd.Series(0, index=data.index)

        if self.similarity_function == "":
            return pd.Series(0, index=data.index)

        model_fn = self.container.model.similarity_functions[self.similarity_function]
        # results = model_fn(data, self.container, self.reference_texts[0], self.text_col)
        results = model_fn(data, self)
        return results

    def to_dict(self) -> dict[str, any]:
        """Get a dictionary of all relevant parameters that define this anchor."""
        params = super().to_dict()
        self_params = dict(similarity_function=self.similarity_function)
        params.update(self_params)
        return params
