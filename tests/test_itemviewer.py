"""Tests for the item viewer - the other tab with the datamanager."""

import pytest

from icat.model import Model


@pytest.mark.integration
def test_populate_called_on_model_trained(mocker, fun_df, dummy_anchor):
    """When the model is trained, the populate() function should be re-called with
    whatever the current instance index is."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    model.data.item_viewer.populate(5)

    spy = mocker.spy(model.data.item_viewer, "populate")

    for i in range(11):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)

        if model.is_seeded() and model.is_trained():
            assert dummy_anchor.name in model.anchor_list.coverage_info

    assert spy.call_count == 11
    assert spy.call_args.args == (5,)


@pytest.mark.integration
def test_clicking_interesting_button_correctly_labels(fun_df):
    """Clicking the interesting button should correctly fire a labeled event
    and the current label should reflect."""
    returns = []

    def catch_label(index, new_label):
        nonlocal returns
        returns.append(index)
        returns.append(new_label)

    model = Model(fun_df, text_col="text")
    model.data.on_data_labeled(catch_label)
    model.data.item_viewer.populate(5)
    model.data.item_viewer.interesting_button.fire_event("click", True)

    assert returns[0] == 5
    assert returns[1] == 1
    assert model.data.item_viewer.current_label.children == ["Labeled"]
    assert model.data.item_viewer.current_label.class_ == "orange--text darken-1"


@pytest.mark.integration
def test_clicking_uninteresting_button_correct_labels(fun_df):
    """Clicking the uninteresting button should correctly fire a labeled event
    and current label should reflect."""
    returns = []

    def catch_label(index, new_label):
        nonlocal returns
        returns.append(index)
        returns.append(new_label)

    model = Model(fun_df, text_col="text")
    model.data.on_data_labeled(catch_label)
    model.data.item_viewer.populate(5)
    model.data.item_viewer.uninteresting_button.fire_event("click", True)

    assert returns[0] == 5
    assert returns[1] == 0
    assert model.data.item_viewer.current_label.children == ["Labeled"]
    assert model.data.item_viewer.current_label.class_ == "blue--text darken-1"
