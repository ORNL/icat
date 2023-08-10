"""Class for a better (specific) table for the datamanager class than tabulator."""

from collections.abc import Callable

import ipyvuetify as v
import traitlets

from icat.utils import _kill_param_auto_docstring

_kill_param_auto_docstring()


class TableContentsTemplate(v.VuetifyTemplate):
    items = traitlets.List(traitlets.Dict()).tag(sync=True)
    headers = traitlets.List(traitlets.Dict()).tag(
        sync=True
    )  # requires at least text, value
    height = traitlets.Int(650).tag(sync=True)
    width = traitlets.Int(450).tag(sync=True)
    total_length = traitlets.Int(0).tag(sync=True)

    options = traitlets.Dict().tag(sync=True)

    def __init__(self, *args, **kwargs):
        self._add_selected_text_callbacks: list[Callable] = []
        self._select_point_callbacks: list[Callable] = []
        self._apply_label_callbacks: list[Callable] = []
        self._update_options_callbacks: list[Callable] = []
        self._hover_point_callbacks: list[Callable] = []
        self._add_example_callbacks: list[Callable] = []

        super().__init__(*args, **kwargs)

        # self.on_msg(lambda widget, content, buffers: print(content))

    #     self._add_selected_text_callbacks: list[Callable]
    #     self._select_point_callbacks: list[Callable]
    #     self._apply_label_callbacks: list[Callable]
    #     self._update_options_callbacks: list[Callable]

    def on_apply_label(self, callback: callable):
        """Expect a point id and a label value (0 or 1)"""
        self._apply_label_callbacks.append(callback)

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

    def vue_addToExampleAnchor(self, point_id):
        for callback in self._add_example_callbacks:
            callback(point_id)

    def vue_updateOptions(self, data):
        self.options = data
        for callback in self._update_options_callbacks:
            callback(data)

    # NOTE: I'm leaving this here for reference, I don't actually think there's any reason to need to go this direction
    # but this is a functioning way to do it (note that this will _retrigger_ the updateOptions handler)
    # def change_page(self, page_num):
    #     new_options = dict(**self.options)
    #     new_options["page"] = page_num
    #     self.options = new_options

    @traitlets.default("template")
    def _template(self):
        return """
        <v-data-table
            class='softhover-table'
            :height="height"
            :width="width"
            :items="items"
            :headers="headers"
            :server-items-length="total_length"
            :options="options"
            @update:options="updateOptions"
        >
            <template #body="{ items }">
                <tbody>
                    <tr v-for="item in items" :key="item.id" @click="selectPoint(item.id)" @mouseover="hoverPoint(item.id)">
                        <td class="break-word" v-html="item.text" />
                        <td style='vertical-align: top;'>{{ item.id }}</td>
                        <td style="vertical-align: top; padding-bottom: 5px">
                            <v-btn x-small class="orange darken-1" @click.stop="applyAbsoluteLabelUninteresting(item.id)">
                                U
                            </v-btn>
                            <v-btn x-small class="blue darken-1" @click.stop="applyAbsoluteLabelInteresting(item.id)">
                                I
                            </v-btn>
                            <v-btn x-small class="purple darken-1" @click.stop="addToExampleAnchor(item.id)">
                                example
                            </v-btn>
                            <div v-html="item.labeled" />
                        </td>
                    </tr>
                </tbody>
            </template>
        </v-data-table>
        <style id='softhover-table-style'>
            .softhover-table table tbody tr:hover {
                background-color: #333333 !important;
            }
            .break-word {
                word-break: break-word;
            }
        </style>
        """
