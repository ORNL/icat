"""Module for the view code that manages interactions between ipyanchorviz/various
widgets and the rest of the model gui components."""

import copy
import random
from collections.abc import Callable
from typing import Any

import ipywidgets as ipw
import pandas as pd
import panel as pn
from ipyanchorviz import AnchorViz

from icat.anchors import Anchor, DictionaryAnchor
from icat.histograms import Histograms
from icat.utils import _kill_param_auto_docstring

_kill_param_auto_docstring()


class InteractiveView(pn.viewable.Viewer):
    """The GUI/widget dashboard for interacting with the model, data, and anchors.

    This class glues all of the events together across the various components (anchorlist,
    datamanager, ipyanchorviz etc.), and itself is the layout container for the full dashboard.

    Args:
        model (icat.Model): The parent model that this view is associated with.
    """

    # TODO: coupling: technically coupling model and view, I think I care a lot less about it
    # here since view is already loosely orchestrating all of the gui stuff. If anything,
    # _more_ of the model event handlers should be handled in here instead of in the model?
    def __init__(self, model, **params):
        self.uninteresting_color = "#2196F3"  # blue
        self.interesting_color = "#FB8C00"  # orange darken-1

        self.model = model

        self.anchorviz: AnchorViz = AnchorViz(
            margin=dict(left=80, top=30, right=80, bottom=30), autoNorm=False
        )

        self.histograms = Histograms()

        self.debug = ipw.Output()

        self._selected_points_change_callbacks: list[Callable] = []

        self.layout = pn.Row(
            pn.Column(self.anchorviz, self.model.anchor_list, self.debug),
            pn.Column(self.model.data.widget, self.histograms, width=700),
            height=1150,
        )

        # set up all of the event handlers
        self.model.anchor_list.on_anchor_added(self._add_list_anchor_to_viz)
        self.model.anchor_list.on_anchor_changed(
            self._send_anchorlist_anchor_modification_to_viz
        )
        self.model.anchor_list.on_anchor_changed(
            self._update_data_table_on_anchor_change
        )
        self.model.anchor_list.on_anchor_added(self._update_data_table_on_anchor_added)
        self.model.anchor_list.on_anchor_removed(self._remove_list_anchor_from_viz)
        self.model.anchor_list.on_anchor_types_changed(
            self._update_viz_anchor_colors_from_type
        )
        self.anchorviz.on_anchor_add(self._add_viz_anchor_to_list)
        self.anchorviz.observe(
            self._trigger_selected_points_change, names="lassoedPointIDs"
        )
        self.model.data.table.on_point_hover(self._set_anchorviz_selected_point)
        self.model.data.on_sample_changed(self._handle_data_sample_changed)
        self.histograms.on_range_changed(self._histograms_range_changed)
        super().__init__(**params)
        self.refresh_data()

    def _handle_data_sample_changed(self, new_sample_indices: list[int]):
        """When the model's data manager sample_indices changes, it fires the
        on_sample_changed event."""
        self.refresh_data()

    def _histograms_range_changed(self, range: list[int]):
        """Limit the set of points displayed in anchorviz to only prediction
        outputs within the specified range. [min, max]"""
        self.model.data.pred_min = range[0]
        self.model.data.pred_max = range[1]
        self.anchorviz.set_points(self._serialize_data_to_dicts())

    def _set_anchorviz_selected_point(self, point_id: int):
        """Set the index of the highlighted red datapoint."""
        self.anchorviz.selectedDataPointID = str(point_id)

    def _add_list_anchor_to_viz(self, anchor: Anchor):
        """Whenever an anchor is added in the anchorlist, add it to
        the visualization as well."""
        # TODO: do some logic to distribute, e.g. find the longest
        # difference between thetas and put the new one in the middle of it.

        # make sure this isn't a double call from when you click on the anchorviz ring
        for av_anchor in self.anchorviz.anchors:
            if av_anchor["id"] == anchor.name:
                return

        # TODO: randomize theta here
        theta = (
            anchor.theta if hasattr(anchor, "theta") else random.uniform(0, 2 * 3.14)
        )
        anchor_dict = dict(id=anchor.name, name=anchor.anchor_name, theta=theta)
        anchor_dict["color"] = self.model.anchor_list.get_anchor_type_config(
            type(anchor)
        )["color"]
        self.anchorviz.add_anchor(anchor_dict)

    def _remove_list_anchor_from_viz(self, anchor: Anchor):
        """When an anchor is removed from the anchorlist, propagate
        that removal to the visualization."""
        # TODO: this can be cleaned when ipyanchorviz#6 is implemented
        current_anchors = copy.deepcopy(self.anchorviz.anchors)
        for anchor_dict in current_anchors:
            if anchor_dict["id"] == anchor.name:
                current_anchors.remove(anchor_dict)

        self.anchorviz.set_anchors(current_anchors)

    def _add_viz_anchor_to_list(self, content: dict):
        """Event handler for when a new anchor was added to the visualization,
        ensure it also gets added to the anchorlist."""
        new_anchor_dict = content["newAnchor"]
        name = self.model.anchor_list.get_unique_anchor_name()
        new_anchor = DictionaryAnchor(
            anchor_name=name, container=self.model.anchor_list
        )
        # self.suppress_next_add_event = True

        # send the actual name to be the id for the visualization
        self.anchorviz.modify_anchor_by_id(new_anchor_dict["id"], "id", new_anchor.name)
        self.anchorviz.modify_anchor_by_id(new_anchor.name, "name", name)
        self.anchorviz.modify_anchor_by_id(
            new_anchor.name,
            "color",
            self.model.anchor_list.get_anchor_type_config(type(new_anchor))["color"],
        )

        self.model.anchor_list.add_anchor(new_anchor)

    def _update_viz_anchor_colors_from_type(
        self, anchor_type_dicts: list[dict[str, any]]
    ):
        for anchor in self.model.anchor_list.anchors:
            if anchor.in_view:
                self.anchorviz.modify_anchor_by_id(
                    anchor.name,
                    "color",
                    self.model.anchor_list.get_anchor_type_config(type(anchor))[
                        "color"
                    ],
                )

    # TODO: need tests for this one
    def _send_anchorlist_anchor_modification_to_viz(
        self, id: str, property: str, value: Any
    ):
        """Whenever a modification is made to an anchor in the anchorlist (here
        we only care about name changes, since that's the only visual change),
        propagate it to the anchorviz visualization."""
        if property == "anchor_name":
            self.anchorviz.modify_anchor_by_id(id, "name", value)
        elif property == "in_view":
            if not value:
                # add theta to the anchor so that when we "re-check" the anchor and re-add it to the viz,
                # it adds it back where it was when we removed it.
                for anchor in self.anchorviz.anchors:
                    if anchor["id"] == id:
                        actual_anchor = self.model.anchor_list.get_anchor_by_panel_id(
                            id
                        )
                        actual_anchor.theta = anchor["theta"]
                # remove the anchor, but only from the visualization.
                # TODO: this can be cleaned when ipyanchorviz#6 is implemented
                updated_anchors = copy.deepcopy(self.anchorviz.anchors)
                for anchor_dict in updated_anchors:
                    if anchor_dict["id"] == id:
                        updated_anchors.remove(anchor_dict)
                self.anchorviz.set_anchors(updated_anchors)
            else:
                actual_anchor = self.model.anchor_list.get_anchor_by_panel_id(id)
                self._add_list_anchor_to_viz(actual_anchor)

    def _serialize_data_to_dicts(self) -> dict:
        feature_names = self.model.feature_names()

        if self.model.data.active_data is None:
            return []

        # first we need to see if we are filtering data based on prediction range
        # we reference the sample indices a lot, so just compute this once and reference
        # throughout.
        sample_in_range_indices = self.model.data.sample_indices
        if self.model.data.prediction_col in self.model.data.active_data.columns:
            sample_in_range_indices = self.model.data._prediction_range_filter(
                self.model.data.active_data.loc[self.model.data.sample_indices]
            ).index

        weights = self.model.data.active_data.loc[
            sample_in_range_indices, feature_names
        ]

        # NOTE: (2/23/2023) previously we were doing normalization directly on
        # the features in the datamanager (as normalized by the anchor_list),
        # but I'm not sure that it's important for the data the model is using
        # to be normalized in this way, it's only important for the
        # visualization.
        weights = weights.apply(
            lambda row: row / row.sum() if row.sum() > 1.0 else row, axis=1
        )

        # change the "feature names" to the associated anchor panel ids
        col_name_changes = {}
        for index, name in enumerate(feature_names):
            col_name_changes[name] = self.model.anchor_list.anchors[index].name
        weights_rows = weights.rename(columns=col_name_changes).to_dict(
            orient="records"
        )

        points = pd.DataFrame(index=weights.index)
        points["id"] = [str(index) for index in points.index]

        # set point color based on prediction
        if not self.model.is_seeded() or not self.model.is_trained():
            points["color"] = "white"
        else:
            points["color"] = self.model.data.active_data.loc[
                sample_in_range_indices, self.model.data.prediction_col
            ].apply(
                lambda val: self.interesting_color
                if val > 0.5
                else self.uninteresting_color
            )

        if self.model.data.label_col in self.model.data.active_data.columns:
            points["labeled"] = self.model.data.active_data.loc[
                sample_in_range_indices, self.model.data.label_col
            ].apply(lambda val: True if val != -1 else False)

        if len(weights_rows) > 0:
            points["weights"] = weights_rows

        return points.to_dict(orient="records")

    def _trigger_selected_points_change(self, change):
        selected_ids = self.anchorviz.lassoedPointIDs
        for callback in self._selected_points_change_callbacks:
            callback(selected_ids)

    def _update_data_table_on_anchor_change(self, id: str, property: str, value: Any):
        """Whenever an anchor changes, refresh the data table (this is so that any
        keyword changes correctly update highlighting)"""
        # NOTE: unsure if this will cause any "jumping" or lag, if so, look for a keywords
        # property?
        self.model.data.update_trigger = True

    def _update_data_table_on_anchor_added(self, anchor: Anchor):
        """Whenever an anchor is added, refresh the data table (this is so that any
        keyword changes correctly update highlighting)"""
        # NOTE: unsure if this will cause any "jumping" or lag, if so, look for a keywords
        # property?
        self.model.data.update_trigger = True

    def on_selected_points_change(self, callback: Callable):
        """Register a callback function for the "anchor added" event.

        Callbacks for this event should take a single parameter which is list of
        selected point IDs (strings most likely).
        """
        self._selected_points_change_callbacks.append(callback)

    def refresh_data(self):
        """Refresh all components with the latest active_data from parent model's ``DataManager``."""
        self.anchorviz.set_points(self._serialize_data_to_dicts())
        self.model.data.item_viewer.populate(self.model.data.item_viewer.index)
        self.histograms.refresh_data(self.model.data)

    def __panel__(self):
        return self.layout
