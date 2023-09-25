"""The model class, this handles both the machine learning model, as well as
contains an associated anchorlist, datamanager, and view. This is sort of the
primary parent class for interacting with icat.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import icat
from icat.anchorlist import AnchorList
from icat.anchors import Anchor, DictionaryAnchor, TFIDFAnchor
from icat.data import DataManager
from icat.utils import _kill_param_auto_docstring
from icat.view import InteractiveView

_kill_param_auto_docstring()


class Model:
    """The interactive machine learning model - a basic binary classifier with tools
    for viewing and interacting with the data and features.

    Args:
        data (pd.DataFrame): The data to explore with.
        text_col (str): The name of the text column in the passed data.
        anchor_types (list[type | dict[str, any]]): The list of class types of \
            anchors to initially include in the interface. (This can be modified \
            after initialization through the ``anchor_list``.)
        default_sample_size (int): The initial number of points to sample for the \
            visualizations.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        text_col: str,
        anchor_types: list[type | dict[str, any]] = None,
        default_sample_size: int = 100,
    ):
        if anchor_types is None:
            anchor_types = [
                DictionaryAnchor,
                {
                    "ref": TFIDFAnchor,
                    "color": icat.anchorlist.ANCHOR_COLOR_PALLETE[0][1],
                },
            ]

        self.training_data: pd.DataFrame = None
        """The rows (and only those rows) of the original data explicitly used for training."""
        self.text_col = text_col
        """The column in the dataframe with the text to explore."""

        self.classifier: LogisticRegression = LogisticRegression(
            class_weight="balanced"
        )
        """The underlying machine learning algorithm that learns based on the training data."""

        self.anchor_list: AnchorList = AnchorList(model=self, anchor_types=anchor_types)
        """The ``AnchorList`` instance that manages all features/featuring necessary for
        the classifier."""
        self.data: DataManager = DataManager(
            data=data,
            text_col=text_col,
            model=self,
            default_sample_size=default_sample_size,
        )
        """The ``DataManager`` instance that handles all labeling tasks and data filtering/sampling."""
        self.view: InteractiveView = InteractiveView(model=self)
        """The ``InteractiveView`` or dashboard widget that glues together the various visual components."""

        # set up necessary behind-the-scenes glue for anchors and data
        self.anchor_list.on_anchor_added(self._on_anchor_add)
        self.anchor_list.on_anchor_removed(self._on_anchor_remove)
        self.anchor_list.on_anchor_changed(self._on_anchor_change)
        self.data.on_data_labeled(self._on_data_label)
        self.view.on_selected_points_change(self._on_selected_points_change)

        self._last_anchor_names: dict[str, str] = []
        """Keep track of anchor names so when the name of one updates we can
        remove the previous column name. The key is the panel id."""

        self.anchor_list.build_tfidf_features()

    def _on_data_label(self, index: int | list[int], new_label: int | list[int]):
        """Event handler for datamanager.

        if a -1 is passed for (or in the list of) new_label, remove it from the
        training data, if found.
        """

        # expand a single value pass to list for consistent handling below
        if type(index) != list:
            index = [index]
            new_label = [new_label]

        if self.training_data is None:
            self.training_data = pd.DataFrame(self.data.active_data.loc[index, :])
        else:
            for i in range(len(index)):
                if (
                    index[i] in self.training_data.index
                    and self.training_data.loc[index[i], self.data.text_col]
                    == self.data.active_data.loc[index[i], self.data.text_col]
                ):
                    self.training_data.at[index[i], self.data.label_col] = new_label[i]
                else:
                    self.training_data = pd.concat(
                        [
                            self.training_data,
                            pd.DataFrame([self.data.active_data.loc[index[i], :]]),
                        ]
                    )

        # Remove any rows from the resulting table where the label_col is "-1", indicating an
        # unlabelling operation
        self.training_data = self.training_data.drop(
            self.training_data[self.training_data[self.data.label_col] == -1].index
        )

        # note that we don't call fit because we don't need to re-featurize, only a label has changed
        self._train_model()
        self.view.refresh_data()

    def _on_anchor_change(self, name: str, property: str, value: any):
        """Event handler for anchorlist."""
        if property == "anchor_name":
            # short-circuit if the name is the same as another anchor - if we continue with
            # the code below, then two _last_anchor_names will have the same value and the
            # columns in the dataframe end up the same and can't be "de-synced" in the interface.
            anchor = self.anchor_list.get_anchor_by_panel_id(name)
            if f"_{value}" in self.data.active_data.columns:
                anchor._anchor_name_input.error = True
                anchor._anchor_name_input.error_messages = (
                    "Another anchor is using this name."
                )
                return
            else:
                anchor._anchor_name_input.error = False
                anchor._anchor_name_input.error_messages = ""

            # no anchor value changes, so no need to re-featurize
            col_name = f"_{self._last_anchor_names[name]}"
            self.data.active_data.rename(columns={col_name: f"_{value}"}, inplace=True)
            if self.training_data is not None:
                self.training_data.rename(columns={col_name: f"_{value}"}, inplace=True)
            self._last_anchor_names[name] = value
        else:
            self.fit()
            self.view.refresh_data()

    def _on_anchor_remove(self, anchor: Anchor):
        """Event handler for anchorlist."""
        col_name = f"_{anchor.anchor_name}"
        if col_name in self.data.active_data.columns:
            self.data.active_data.drop(columns=[col_name], inplace=True)
        if self.training_data is not None and col_name in self.training_data.columns:
            self.training_data.drop(columns=[col_name], inplace=True)
        self.fit()
        self.view.refresh_data()

    def _on_anchor_add(self, anchor: Anchor):
        """Event handler for anchorlist."""
        # TODO: it's possible this should be handled inside the anchorlist?
        if hasattr(anchor, "text_col") and anchor.text_col == "":
            anchor.text_col = self.text_col
            self.anchor_list.anchors[-1].text_col = self.text_col
        self.fit()
        self.view.refresh_data()

    # TODO: I wonder if this should directly be in the view?
    def _on_selected_points_change(self, selected_ids: list[str]):
        """Event handler for interactive view."""
        self.data.selected_indices = [int(selected_id) for selected_id in selected_ids]

    def _train_model(self):
        """Fits the data to the current training dataset, note that this function
        assumes the data has already been featurized."""

        if not self.is_seeded():
            # we short circuit training the model here, but we do still want to show
            # coverage stats even if it's not seeded, so compute and set here
            coverage_info = self.compute_coverage()
            self.anchor_list.set_coverage(coverage_info)

            # if this model _was_ trained but we unlabel sufficiently many points, we need to consider
            # it untrained
            if self.is_trained():
                del self.classifier.classes_
                self.data.active_data = self.data.active_data.drop(
                    self.data.prediction_col, axis=1
                )
            return False

        if len(self.feature_names(in_model_only=True)) < 1:
            return False
        self.classifier.fit(
            self.training_data[self.feature_names(in_model_only=True)],
            self.training_data[self.data.label_col],
        )
        self.data.active_data[self.data.prediction_col] = self.predict()
        coverage_info = self.compute_coverage()
        self.anchor_list.set_coverage(coverage_info)

    def compute_coverage(self) -> dict[str, dict[str, float | int]]:
        """Calculate the coverage of the current anchors on the current active data.

        Returns:
            A dictionary where each key is the panel id of the anchor, and the value
            is a dictionary with the statistics: ``'total'``, ``'pos'``, ``'neg'``,
            ``'total_pct'``, ``'pos_pct'``, and ``'neg_pct'``
        """
        features = self.data.active_data.loc[:, self.feature_names()].values
        predictions = (
            self.data.active_data[self.data.prediction_col].values
            if self.data.prediction_col in self.data.active_data
            else None
        )

        total_count = np.ceil(features).clip(0, 1).sum(axis=0)
        total_pct = total_count / features.shape[0]

        if predictions is not None:
            positive_indices = np.where(predictions >= 0.5)
            pos_count = np.ceil(features[positive_indices]).clip(0, 1).sum(axis=0)
            pos_pct = pos_count / total_count
            pos_pct = np.nan_to_num(pos_pct, nan=0.0)

            negative_indices = np.where(predictions < 0.5)
            neg_count = np.ceil(features[negative_indices]).clip(0, 1).sum(axis=0)
            neg_pct = neg_count / total_count
            neg_pct = np.nan_to_num(neg_pct, nan=0.0)
        else:
            pos_count = None
            pos_pct = None
            neg_count = None
            neg_pct = None

        coverage_info = {}
        # NOTE: we can simply go through by index because feature_names is also
        # iterating over self.anchor_list.anchors
        for index, anchor in enumerate(self.anchor_list.anchors):
            coverage_info[anchor.name] = {
                "total": total_count[index],
                "total_pct": total_pct[index],
                "pos": pos_count[index] if pos_count is not None else 0.0,
                "pos_pct": pos_pct[index] if pos_pct is not None else 0.0,
                "neg": neg_count[index] if neg_count is not None else 0.0,
                "neg_pct": neg_pct[index] if neg_pct is not None else 0.0,
            }
        return coverage_info

    def is_seeded(self) -> bool:
        """Determine if there are enough labels in the training data to train the model with.

        Returns:
            False if the label column doesn't exist, there's fewer than 10 labeled points,
            or there's only one class of label.
        """
        if self.training_data is None or self.data.label_col not in self.training_data:
            # no labels!
            return False
        labeled_df = self.training_data[self.training_data[self.data.label_col] != -1]
        if len(labeled_df) < 10:
            # Not enough labels
            return False
        if len(labeled_df[self.data.label_col].value_counts()) == 1:
            # No diversity in labels
            return False
        return True

    def is_trained(self) -> bool:
        return hasattr(self.classifier, "classes_")

    def featurize(
        self,
        data: pd.DataFrame | None = None,
        normalize: bool = False,
        normalize_reference: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run the anchors - calculates the output features for each anchor and adds the corresponding "weights" column
        to the dataframe. These are the values that the classifier uses to make its predictions.

        Args:
            data (pd.DataFrame): The data to apply the anchors to. Uses the exploration data if not specified.
            normalize (bool): Whether to apply l1 normalization to the output values.
            normalize_reference (Optional[pd.DataFrame]): A different dataframe whose features to sum for the L1 norm, this \
                is used with the model's separate training data versus full dataset, since the normed values of just the \
                training data would be vastly different than within the full set.

        Returns:
            The passed data with the feature columns on it.
        """
        # NOTE: normalize here is normalizing along a column
        if data is None:
            data = self.data.active_data
        data = self.anchor_list.featurize(data, normalize, normalize_reference)
        return data

    def fit(self):
        """Featurize the current data and fit the model to it."""
        # self.data.active_data = self.featurize(self.data.active_data, True, None)
        # self.training_data = self.featurize(
        #     self.training_data, True, self.data.active_data
        # )

        # Since training data needs to be normalized based on _raw_ updated active data values,
        # we featurize active data _without_ normalizing. (Norming based on already normed values
        # gives incorrect results)
        features = self.feature_names()
        self._last_anchor_names = {
            anchor.name: anchor.anchor_name for anchor in self.anchor_list.anchors
        }
        if len(features) == 0:
            # we can't train if we have no features!
            return

        # self.norm_reference = self.featurize(self.data.active_data, normalize=False)[
        #     features
        # ].copy()
        self.data.active_data = self.featurize(self.data.active_data, normalize=False)
        if self.training_data is not None:
            self.training_data = self.featurize(
                self.training_data,
                normalize=False,  # NOTE: (2/8/23) taking out norm for now because broken/unclear if necessary
                # normalize_reference=self.norm_reference,
            )
        # Now we can finish the normalization of the active data.
        # NOTE: (2/8/23) taking out normalization for now because it's currently broken/unclear if necessary.
        # self.data.active_data.loc[:, features] = self.data.active_data[features].apply(
        #     AnchorList._l1_col_normalize, axis=0
        # )

        self._train_model()

    def feature_names(self, in_model_only: bool = False) -> list[str]:
        """Provides a list of the feature column names in use in the data manager.

        Args:
            in_model_only (bool): Only include anchors whose ``in_model`` value is \
                ``True``.
        """
        # TODO: conditional for if it's enabled in model?
        if in_model_only:
            names = [
                f"_{anchor.anchor_name}"
                for anchor in self.anchor_list.anchors
                if anchor.in_model
            ]
        else:
            names = [f"_{anchor.anchor_name}" for anchor in self.anchor_list.anchors]
        return names

    # NOTE: if using datamanager's data, we're assuming we've already fit and featurized
    def predict(
        self, data: pd.DataFrame | None = None, inplace: bool = True
    ) -> np.ndarray:
        """Run model's classifier predictions on either the passed data or training data.

        Note:
            This function, like sklearn, assumes the model has already been fit.
            (We have no strict check for this, as this is for IML and the classifier is assumed
            to be re-fit multiple times.)

        Args:
            data (Optional[pd.DataFrame]): If not specified, use the previously set training data, \
                otherwise predict on this data.
            inplace (bool): Whether to operate directly on the passed data or create a copy of it.

        Returns:
            The predictions for either the active or passed data if provided.
        """
        if data is not None:
            if not inplace:
                data = data.copy(True)
            self.featurize(
                data,  # normalize=True, normalize_reference=self.norm_reference
                # NOTE: (2/8/23) turning off normalization for now until figure out assumptions
            )
        else:
            data = self.data.active_data

        # compute and return probability of class 1, so range 0-1 is directly correspondent
        predictions = self.classifier.predict_proba(
            data[self.feature_names(in_model_only=True)]
        )[:, 1]

        return predictions

    def add_anchor(self, anchor: Anchor):
        """Add the passed anchor to this model's anchor list.

        Args:
            anchor (Anchor): The Anchor to add to the list.

        Note:
            See ``AnchorList.add_anchor`` for more details.
        """
        self.anchor_list.add_anchor(anchor)

    def save(self, path: str):
        """Save the model and all associated data at the specified location."""

        os.makedirs(f"{path}", exist_ok=True)

        # save the anchorlist
        self.anchor_list.save(path)

        # save any relevant model metadata
        model_information = {
            "icat_version": icat.__version__,
            "timestamp": datetime.strftime(datetime.now(), "%m/%d/%y %H:%M:%S"),
            "text_col": self.text_col,
            "sample_size": self.data.sample_size_txt.v_model,
        }
        with open(f"{path}/model_info.json", "w") as outfile:
            json.dump(model_information, outfile)

        # save a dump of the training data
        if self.training_data is not None:
            self.training_data.to_pickle(f"{path}/training_data.pkl")

        # save a dump of the active data in data manager
        # TODO: maybe this isn't necessary?
        self.data.active_data.to_pickle(f"{path}/active_data.pkl")

        # save the sample set
        with open(f"{path}/active_data_sample.json", "w") as outfile:
            json.dump(self.data.sample_indices, outfile)

        # save the classifier
        joblib.dump(self.classifier, f"{path}/classifier.joblib")

    @staticmethod
    def load(path: str) -> "Model":
        """Reload the model with all of the data and anchors from the specified location.

        Example:

            .. code-block:: python

                import icat

                m1 = icat.Model(my_data, text_col="text")
                m1.save("~/tmp/my_model")

                m2 = icat.Model.load("~/tmp/my_model")
        """

        # load relevant model metadata
        with open(f"{path}/model_info.json") as infile:
            model_information = json.load(infile)

        # version check
        saved_major, saved_minor, saved_patch = (
            int(i) for i in model_information["icat_version"].split(".")
        )
        major, minor, patch = (int(i) for i in icat.__version__.split("."))
        if major != saved_major or (saved_minor < 7 and saved_major < 1):
            print("ERROR - Model was saved with incompatible version of icat")
            return None

        data = pd.read_pickle(f"{path}/active_data.pkl")

        # init with no anchor types because anchor_list loading will handle that
        model = Model(
            data=data, anchor_types=[], text_col=model_information["text_col"]
        )
        model.data.sample_size_txt.v_model = model_information["sample_size"]

        # load the classifier
        model.classifier = joblib.load(f"{path}/classifier.joblib")

        # load the training data
        if os.path.exists(f"{path}/training_data.pkl"):
            model.training_data = pd.read_pickle(f"{path}/training_data.pkl")

        # load the sample set into the data manager
        with open(f"{path}/active_data_sample.json") as infile:
            model.data.sample_indices = json.load(infile)

        # load the anchorlist
        model.anchor_list.load(path)

        return model
