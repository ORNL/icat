"""Class for a better (specific) table for the datamanager class than tabulator."""

from collections.abc import Callable

import ipyvuetify as v
import traitlets

from icat.utils import _kill_param_auto_docstring

_kill_param_auto_docstring()


class TableContentsTemplate(v.VuetifyTemplate):
    """The primary table used in the ``DataManager`` class. This is a heavily modified vuetify
    DataTable, largely necessary for us to be able to put a more comprehensive set of action
    buttons in each row."""

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
            dense
            :items="items"
            :headers="headers"
            :server-items-length="total_length"
            :options="options"
            @update:options="updateOptions"
        >
            <template #body="{ items }">
                <tbody :width="width">
                    <tr v-for="item in items" :key="item.id" @click="selectPoint(item.id)" @mouseover="hoverPoint(item.id)">
                        <td v-html="item.text" style="padding-left: 2px; padding-right: 2px; word-break: break-word;" />
                        <td style="vertical-align: top; padding-left: 2px; padding-right: 2px; color: grey;">{{ item.id }}</td>
                        <td style="vertical-align: top; padding-bottom: 5px; padding-left: 2px;">
                            <v-tooltip bottom open-delay=500>
                                <template v-slot:activator="{ on, attrs }">
                                    <v-btn x-small class="blue darken-1" @click.stop="applyAbsoluteLabelUninteresting(item.id)" v-bind="attrs" v-on="on">
                                        U
                                    </v-btn>
                                </template>
                                <span>Label this item as <span class="blue--text lighten-4"><b>uninteresting</b></span> (cold).</span>
                            </v-tooltip>
                            <v-tooltip bottom open-delay=500>
                                <template v-slot:activator="{ on, attrs }">
                                    <v-btn x-small class="orange darken-1" @click.stop="applyAbsoluteLabelInteresting(item.id)" v-bind="attrs" v-on="on">
                                        I
                                    </v-btn>
                                </template>
                                <span>Label this item as <span class="orange--text lighten-5"><b>interesting</b></span> (warm).</span>
                            </v-tooltip>
                            <v-tooltip bottom open-delay=500 v-if="example_btn_color != ''">
                                <template v-slot:activator="{ on, attrs }">
                                    <v-btn x-small :style="{ backgroundColor: example_btn_color }" @click.stop="addToExampleAnchor(item.id)" v-bind="attrs" v-on="on">
                                        example
                                    </v-btn>
                                </template>
                                <span>Create a {{ example_type_name }} anchor with this item as the target.</span>
                            </v-tooltip>
                            <v-tooltip bottom open-delay=500>
                                <template v-slot:activator="{ on, attrs }">
                                    <v-btn x-small v-if="!item.in_sample" @click.stop="addToSample(item.id)" v-bind="attrs" v-on="on">
                                        sample
                                    </v-btn>
                                </template>
                                <span>Add this item to the current sample set.</span>
                            </v-tooltip>
                            <v-tooltip bottom open-delay=500>
                                <template v-slot:activator="{ on, attrs }">
                                    <div v-html="item.labeled" v-bind="attrs" v-on="on" />
                                </template>
                                <span v-if="item.labeled.indexOf('orange') > -1"><span class="orange--text lighten-5"><b>interesting</b></span> (warm).</span>
                                <span v-if="item.labeled.indexOf('blue') > -1"><span class="blue--text lighten-4"><b>uninteresting</b></span> (cold).</span>
                            </v-tooltip>
                            <v-tooltip bottom open-delay=500 v-if="item.labeled != ''">
                                <template v-slot:activator="{ on, attrs }">
                                    <v-btn x-small class="red darken-4" @click.stop="applyAbsoluteLabelUnlabeled(item.id)" v-bind="attrs" v-on="on">
                                        unlabel
                                    </v-btn>
                                </template>
                                <span>Remove the label from this item.</span>
                            </v-tooltip>
                        </td>
                    </tr>
                </tbody>
            </template>
        </v-data-table>
        <style id='softhover-table-style'>
            .softhover-table table tbody tr:hover {
                background-color: #333333 !important;
            }
            .softhover-table table thead tr th {
                padding-left: 5px !important;
                padding-right: 5px !important;
            }
            .v-data-table__wrapper {
                overscroll-behavior: contain !important;
            }
        </style>
        """
