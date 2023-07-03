"""The model class, this handles both the machine learning model, as well as
contains an associated anchorlist, datamanager, and view. This is sort of the
primary parent class for interacting with icat.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from icat.anchorlist import AnchorList
from icat.anchors import Anchor, DictionaryAnchor, SimilarityFunctionAnchor, TFIDFAnchor
from icat.data import DataManager
from icat.view import InteractiveView


class Model:
    def __init__(
        self,
        data: pd.DataFrame,
        text_col: str,
        similarity_functions: list[Callable] = [],
    ):
        self.training_data: pd.DataFrame = None
        """The rows (and only those rows) of the original data explicitly used for training."""
        self.text_col = text_col

        self.classifier: LogisticRegression = LogisticRegression(
            class_weight="balanced"
        )

        self.anchor_list: AnchorList = AnchorList(model=self)
        self.data: DataManager = DataManager(data=data, text_col=text_col, model=self)
        self.view: InteractiveView = InteractiveView(model=self)

        # set up necessary behind-the-scenes glue for anchors and data
        self.anchor_list.on_anchor_added(self._on_anchor_add)
        self.anchor_list.on_anchor_removed(self._on_anchor_remove)
        self.anchor_list.on_anchor_changed(self._on_anchor_change)
        self.data.on_data_labeled(self._on_data_label)
        self.view.on_selected_points_change(self._on_selected_points_change)

        self._last_anchor_names: dict[str, str] = []
        """Keep track of anchor names so when the name of one updates we can
        remove the previous column name. The key is the panel id."""

        # self.similarity_functions = similarity_functions
        self.similarity_functions: dict[str, Callable] = {
            str(func.__name__): func for func in similarity_functions
        }

        self.anchor_list.build_tfidf_features()

    def _on_data_label(self, index: int, new_label: int):
        """Event handler for datamanager."""
        if self.training_data is None:
            self.training_data = pd.DataFrame([self.data.active_data.loc[index, :]])

        # update if it's a row that's already in the training data
        elif (
            index in self.training_data.index
            and self.training_data.loc[index, self.data.text_col]
            == self.data.active_data.loc[index, self.data.text_col]
        ):
            self.training_data.at[index, self.data.label_col] = new_label

        else:
            self.training_data = pd.concat(
                # self.training_data, pd.DataFrame(self.data.active_data[index, :])
                [
                    self.training_data,
                    pd.DataFrame([self.data.active_data.loc[index, :]]),
                ]
            )
        # note that we don't call fit because we don't need to re-featurize, only a label has changed
        self._train_model()
        self.view.refresh_data()

    def _on_anchor_change(self, name: str, property: str, value: any):
        """Event handler for anchorlist."""
        if property == "anchor_name":
            # no anchor value changes, so no need to re-featurize
            col_name = f"_{self._last_anchor_names[name]}"
            self.data.active_data.rename(columns={col_name: f"_{value}"}, inplace=True)
            if self.training_data is not None:
                self.training_data.rename(columns={col_name: f"_{value}"}, inplace=True)
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
        if (
            type(anchor) == DictionaryAnchor
            or type(anchor) == TFIDFAnchor
            or type(anchor) == SimilarityFunctionAnchor
        ) and anchor.text_col == "":
            anchor.text_col = self.text_col
            self.anchor_list.anchors[-1].text_col = self.text_col
        if type(anchor) == SimilarityFunctionAnchor:
            anchor._populate_items()
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
            is a dictionary with the statistics: 'total', 'pos', 'neg', 'total_pct',
            'pos_pct', and 'neg_pct'
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
        to the dataframe. These are the values that the classifier uses.

        Args:
            data (pd.DataFrame): The data to apply the anchors to. Uses the exploration data if not specified.
            normalize (bool): Whether to apply l1 normalization to the output values.
            normalize_reference (Optional[pd.DataFrame]): A different dataframe whose features to sum for the L1 norm, this
                is used with the model's separate training data versus full dataset, since the normed values of just the
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
            in_model_only (bool): Only include anchors whose ``in_model`` value is
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
            data (Optional[pd.DataFrame]): If not specified, use the previously set training data,
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
