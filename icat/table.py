"""Class for a better (specific) table for the datamanager class than tabulator."""

from collections.abc import Callable

import ipyvuetify as v
import traitlets

from icat.utils import _kill_param_auto_docstring, vue_template_path

_kill_param_auto_docstring()


class TableContentsTemplate(v.VuetifyTemplate):
    """The primary table used in the ``DataManager`` class. This is a heavily modified vuetify
    DataTable, largely necessary for us to be able to put a more comprehensive set of action
    buttons in each row."""

    template_file = vue_template_path("manager-table.vue")

    items = traitlets.List(traitlets.Dict()).tag(sync=True)
    headers = traitlets.List(traitlets.Dict()).tag(
        sync=True
    )  # requires at least text, value
    height = traitlets.Int(650).tag(sync=True)
    width = traitlets.Int(450).tag(sync=True)
    total_length = traitlets.Int(0).tag(sync=True)

    options = traitlets.Dict().tag(sync=True)

    example_btn_color = traitlets.Unicode("").tag(sync=True)
    example_type_name = traitlets.Unicode("similarity").tag(sync=True)
    # example_btn_color = traitlets.Unicode("#FFEE22").tag(sync=True)

    def __init__(self, *args, **kwargs):
        self._add_selected_text_callbacks: list[Callable] = []
        self._select_point_callbacks: list[Callable] = []
        self._apply_label_callbacks: list[Callable] = []
        self._add_to_sample_callbacks: list[Callable] = []
        self._update_options_callbacks: list[Callable] = []
        self._hover_point_callbacks: list[Callable] = []
        self._add_example_callbacks: list[Callable] = []

        super().__init__(*args, **kwargs)

    def on_apply_label(self, callback: callable):
        """Expect a point id and a label value (0 or 1)"""
        self._apply_label_callbacks.append(callback)

    def on_add_to_sample(self, callback: callable):
        """Expect a point id"""
        self._add_to_sample_callbacks.append(callback)

    def on_update_options(self, callback: callable):
        """Expect a dictionary with:
            * page: int
            * itemsPerPage: int
            * sortBy: list[str]
            * sortDesc: list[bool]
            * groupBy: list[str]
            * groupDesc: list[bool]
            * multiSort: bool
            * mustSort: bool

        so this will include page change events
        """
        self._update_options_callbacks.append(callback)

    def on_select_point(self, callback: callable):
        """Expect a point ID"""
        self._select_point_callbacks.append(callback)

    def on_point_hover(self, callback: callable):
        """Expect a point ID"""
        self._hover_point_callbacks.append(callback)

    def on_add_example(self, callback: callable):
        """Expect a point ID"""
        self._add_example_callbacks.append(callback)

    def vue_addSelectedText(self, data):
        # TODO: eventually this will handle gathering the user-highlighted text
        pass

    def vue_hoverPoint(self, point_id):
        for callback in self._hover_point_callbacks:
            callback(point_id)

    def vue_selectPoint(self, point_id):
        for callback in self._select_point_callbacks:
            callback(point_id)

    def vue_applyAbsoluteLabelUninteresting(self, point_id):
        for callback in self._apply_label_callbacks:
            callback(point_id, 0)

    def vue_applyAbsoluteLabelInteresting(self, point_id):
        for callback in self._apply_label_callbacks:
            callback(point_id, 1)

    def vue_applyAbsoluteLabelUnlabeled(self, point_id):
        for callback in self._apply_label_callbacks:
            callback(point_id, -1)

    def vue_addToExampleAnchor(self, point_id):
        for callback in self._add_example_callbacks:
            callback(point_id)

    def vue_updateOptions(self, data):
        self.options = data
        for callback in self._update_options_callbacks:
            callback(data)

    def vue_addToSample(self, point_id):
        for callback in self._add_to_sample_callbacks:
            callback(point_id)
