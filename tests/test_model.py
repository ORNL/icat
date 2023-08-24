import pytest

from icat.anchors import DictionaryAnchor
from icat.model import Model


# @pytest.mark.integration
@pytest.mark.parametrize(
    "normalize, expected_vals",
    [
        (False, [0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0]),
        (True, [0, 0, 0, 0, 2 / 5, 0, 2 / 5, 1 / 5, 0, 0, 0, 0]),
    ],
)
def test_model_featurize(
    dummy_anchor, fun_df, normalize: bool, expected_vals: list[int | float]
):
    """Calling featurize on the model should use its data manager's active data, compute
    using the provided anchor, and normalize appropriately."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)

    result = model.featurize(normalize=normalize)
    assert "_test" in result.columns
    assert result["_test"].tolist() == expected_vals


def test_model_featurize_with_reference(dummy_anchor, fun_df):
    """Featurizing a dataset and norming based off of another should use the other's feature
    sum in the norm process."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    model.training_data = fun_df.iloc[3:5, :]

    reference = model.featurize(fun_df, normalize=False)
    result = model.featurize(
        model.training_data, normalize=True, normalize_reference=reference
    )
    assert "_test" in result.columns
    assert result["_test"].tolist() == [0.0, 2 / 5]


# NOTE: marking this as skip until I figure out actual assumptions about how norm should work
@pytest.mark.skip
def test_model_featurize_with_model_norm_reference(dummy_anchor, fun_df):
    """Featurizing a dataset and norming based off its own calculated norm reference should
    use the other's feature sum in the norm process."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    model.training_data = fun_df.iloc[3:5, :]

    # NOTE: model.norm_reference is created from calling fit() from within the add_anchor handler.
    result = model.featurize(
        model.training_data, normalize=True, normalize_reference=model.norm_reference
    )
    assert "_test" in result.columns
    assert result["_test"].tolist() == [0.0, 2 / 5]


# NOTE: marking this as skip until I figure out actual assumptions about how norm should work
@pytest.mark.skip
def test_model_fit_featurizes_correctly(dummy_anchor, fun_df):
    """Calling fit on the model should correctly featurize both training_df and active data."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    model.training_data = fun_df.iloc[3:5, :]
    model.fit()
    assert model.training_data["_test"].tolist() == [0.0, 2 / 5]
    assert model.data.active_data["_test"].tolist() == [
        0,
        0,
        0,
        0,
        2 / 5,
        0,
        2 / 5,
        1 / 5,
        0,
        0,
        0,
        0,
    ]


@pytest.mark.parametrize(
    "labels, expected_result",
    [
        ([-1] * 12, False),
        ([-1] * 11 + [0], False),
        ([-1] * 10 + [0, 1], False),
        ([0] * 10 + [-1, -1], False),
        ([0] * 12, False),
        ([0] * 5 + [1] * 5 + [-1, -1], True),
        ([0] * 10 + [1] * 2, True),
    ],
)
def test_is_seeded(fun_df, labels, expected_result):
    """Is_seeded should only return true if there's at least 10 labels that represent more
    than just one class."""
    model = Model(fun_df, text_col="text")
    fun_df[model.data.label_col] = labels
    model.training_data = fun_df
    assert model.is_seeded() == expected_result


def test_label_event_handler_with_no_training_data_does_not_crash(fun_df):
    """The data labelling event handler _should not error_ if you label something but your
    training dataframe is still None."""
    model = Model(fun_df, text_col="text")
    assert model.training_data is None
    model.data.apply_label(0, 0)
    # model._on_data_label(0, 0)
    assert model.training_data is not None


# TODO: test that labelling correctly adds to training data and TODO: updates
def test_label_adds_to_training_data(fun_df):
    """Labelling two datapoints should add two entries to training data."""
    model = Model(fun_df, text_col="text")
    model.data.apply_label(0, 1)
    model.data.apply_label(2, 0)
    assert len(model.training_data) == 2
    assert model.training_data[model.data.label_col].tolist() == [1, 0]
    assert model.training_data.index.tolist() == [0, 2]


# TODO: test that labelling correctly adds to training data and TODO: updates
def test_labelling_same_row_updates_training(fun_df):
    """Labelling the same datapoint twice should update the original entry in the
    training data."""
    model = Model(fun_df, text_col="text")
    model.data.apply_label(0, 1)
    model.data.apply_label(0, 0)
    assert len(model.training_data) == 1
    assert model.training_data[model.data.label_col][0] == 0


# TODO: tests for on_anchor_remove/add/change etc., will need to use mock for those?


def test_removing_anchor_removes_feature_column(fun_df, dummy_anchor):
    """After deleting an anchor, the active data should no longer have its
    feature column."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    assert "_test" in model.data.active_data.columns

    model.anchor_list.remove_anchor(dummy_anchor)
    assert "_test" not in model.data.active_data.columns


def test_renaming_anchor_removes_previous_feature_column(fun_df, dummy_anchor):
    """When an anchor's name is changed, the previous named feature column shouldn't
    be there anymore."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    assert "_test" in model.data.active_data.columns

    model.anchor_list.anchors[0].anchor_name = "toast"
    assert "_test" not in model.data.active_data.columns
    assert "_toast" in model.data.active_data.columns


def test_labeling_10_points_trains_model(fun_df, dummy_anchor):
    """After labeling the first 10 points, the model should have trained."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    assert not hasattr(model.classifier, "classes_")
    for i in range(11):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)
    assert hasattr(model.classifier, "classes_")


def test_labeling_10_points_doesnot_train_model_if_no_features(fun_df):
    """After labeling the first 10 points, if no anchors were added, the model should
    not have trained and importantly not crashed."""
    model = Model(fun_df, text_col="text")
    assert not hasattr(model.classifier, "classes_")
    for i in range(11):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)
    assert not hasattr(model.classifier, "classes_")


def test_coverage_vals_correct_without_predictions(fun_df):
    """Coverage values for an anchor should accurately reflect the number of instances
    in which its keywords appear. This should work regardless of if predictions exist.
    """
    model = Model(fun_df, text_col="text")

    anchor_1 = DictionaryAnchor(model, anchor_name="anchor1", keywords=["I'm"])  # 2
    anchor_2 = DictionaryAnchor(
        model, anchor_name="anchor2", keywords=["I'm", "muffin"]
    )  # 3

    model.anchor_list.add_anchor(anchor_1)
    model.anchor_list.add_anchor(anchor_2)

    model.featurize()
    coverage = model.compute_coverage()

    assert coverage[anchor_1.name]["total"] == 2
    assert coverage[anchor_2.name]["total"] == 3
    assert coverage[anchor_1.name]["total_pct"] == 2 / 12
    assert coverage[anchor_2.name]["total_pct"] == 3 / 12


def test_coverage_pos_neg_correct(fun_df, dummy_anchor):
    """Coverage and pos/neg determinations should correctly reflect prediction split
    per anchor on the active data."""

    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    model.featurize()
    model.data.active_data[model.data.prediction_col] = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.9,
        0.9,
        0.1,
        0.9,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    coverage = model.compute_coverage()

    assert coverage[dummy_anchor.name]["total"] == 3
    assert coverage[dummy_anchor.name]["total_pct"] == 3 / 12
    assert coverage[dummy_anchor.name]["pos"] == 2
    assert coverage[dummy_anchor.name]["pos_pct"] == 2 / 3
    assert coverage[dummy_anchor.name]["neg"] == 1
    assert coverage[dummy_anchor.name]["neg_pct"] == 1 / 3


def test_training_model_sets_anchorlist_coverage(fun_df, dummy_anchor):
    """Training a model should set the coverage values on the anchorlist."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    for i in range(11):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)

        if model.is_seeded() and model.is_trained():
            assert dummy_anchor.name in model.anchor_list.coverage_info
            assert model.anchor_list.coverage_info[dummy_anchor.name]["pos_pct"] != 0.0
            assert model.anchor_list.coverage_info[dummy_anchor.name]["neg_pct"] == 0.0

    """A non in_model anchor should not be used in the model's features during training."""


@pytest.mark.integration
def test_changing_anchor_name_twice_before_model_trained_modifies_data(
    fun_df, dummy_anchor
):
    """Changing an anchor name when the fit() function isn't being called should appropriately
    change model's _last_anchor_names and thus propagate into the data active_data column.
    """

    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    assert "_test" in model.data.active_data.columns

    model.anchor_list.anchors[0]._anchor_name_input.v_model = "testing"
    model.anchor_list.anchors[0]._anchor_name_input.fire_event("blur", "testing")
    assert "_testing" in model.data.active_data.columns
    assert "_test" not in model.data.active_data.columns

    model.anchor_list.anchors[0]._anchor_name_input.v_model = "testing123"
    model.anchor_list.anchors[0]._anchor_name_input.fire_event("blur", "testing123")
    assert "_testing123" in model.data.active_data.columns
    assert "_testing" not in model.data.active_data.columns


def test_save_load_model(data_file_loc, fun_df, dummy_anchor):
    """Saving a model and then reloading it should load in all the same data and anchors."""

    model = Model(fun_df, text_col="text", default_sample_size=200)
    model.anchor_list.add_anchor(dummy_anchor)
    model.data.apply_label(0, 1)
    model.save(data_file_loc)

    model2 = Model.load(data_file_loc)
    assert len(model2.data.active_data) == 12
    assert model2.text_col == "text"
    assert len(model2.anchor_list.anchors) == 1
    assert len(model2.training_data) == 1
    assert model2.data.sample_size_txt.v_model == "200"


def test_unlabel_removes_from_training_set(fun_df, dummy_anchor):
    """Labeling a point and then unlabeling it should remove it from the training set."""
    model = Model(fun_df, text_col="text")
    model.add_anchor(dummy_anchor)
    model.data.apply_label(1, 1)
    assert len(model.training_data) == 1
    model.data.apply_label(1, -1)
    assert len(model.training_data) == 0


def test_unlabel_when_no_corresp_row_does_not_break(fun_df, dummy_anchor):
    """Specifying to 'unlabel' a row that isn't in the training set shouldn't throw
    an error, the training data should simply still not contain that row."""
    model = Model(fun_df, text_col="text")
    model.add_anchor(dummy_anchor)
    model.data.apply_label(1, -1)
    assert len(model.training_data) == 0


def test_unlabel_after_seed_correctly_unseeds(fun_df, dummy_anchor):
    """Unlabeling a point immediately after a model is seeded should effectively
    unseed the model without causing an error."""
    model = Model(fun_df, text_col="text")
    model.add_anchor(dummy_anchor)
    for i in range(0, 10):
        if i in [3, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)
    assert model.is_seeded()
    assert model.is_trained()
    model.data.apply_label(7, -1)
    assert not model.is_seeded()
    assert not model.is_trained()


def test_unlabel_multi(fun_df, dummy_anchor):
    """Calling label function with multiple -1's should correctly unlabel
    multiple points."""
    model = Model(fun_df, text_col="text")
    model.add_anchor(dummy_anchor)
    model.data.apply_label([1, 2], [1, 1])
    assert len(model.training_data) == 2
    model.data.apply_label([1, 2], [-1, -1])
    assert len(model.training_data) == 0


def test_untraining_removes_pred_col(fun_df, dummy_anchor):
    """When a model 'untrains' because the label count fell beneath seeding requirements,
    also remove any previous predictions from the active_data"""

    model = Model(fun_df, text_col="text")
    model.add_anchor(dummy_anchor)
    for i in range(0, 10):
        if i in [3, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)
    assert model.is_seeded()
    assert model.is_trained()
    model.data.apply_label(7, -1)
    assert not model.is_trained()
    assert model.data.prediction_col not in model.data.active_data.columns
