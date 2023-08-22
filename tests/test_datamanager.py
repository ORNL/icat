# TODO: test that a random sample is selected?


import pytest

from icat import Model
from icat.data import DataManager


def test_manager_selected_indices_effects_selected_tab(dummy_data_manager):
    """Setting the selected indices on a manager should change what's in view in the table"""
    dummy_data_manager.current_data_tab = "Selected"
    assert len(dummy_data_manager.filtered_df) == 0
    for i in range(10):
        assert len(dummy_data_manager.table.items) == 0

    dummy_data_manager.selected_indices = [1, 2]
    for i in range(2):
        assert len(dummy_data_manager.table.items) == 2
    assert len(dummy_data_manager.filtered_df) == 2
    assert dummy_data_manager.table.items[0]["text"] == "<span>I like trains</span>"
    assert dummy_data_manager.table.items[1]["text"] == "<span>No llama, no!</span>"


def test_manager_with_no_labels_or_predictions_doesnt_crash(dummy_data_manager):
    """Switching around the various tabs shouldn't crash even if there are no labels or predictions"""
    dummy_data_manager.current_data_tab = "Sample"
    dummy_data_manager.current_data_tab = "Labeled"
    assert dummy_data_manager.current_data_tab == "Labeled"
    assert dummy_data_manager.data_tabs.v_model == 1
    assert len(dummy_data_manager.filtered_df) == 0
    for i in range(10):
        assert len(dummy_data_manager.table.items) == 0


@pytest.mark.parametrize(
    "search_box, first_index, first_text, shape, expect_result",
    [
        ("trains", 1, "I like trains", (1,), True),
        ("", 0, "They said I could never teach a llama to drive!", (12,), True),
        (" ", 0, "They said I could never teach a llama to drive!", (12,), True),
        ("elephant", None, None, (0,), False),
    ],
)
def test_search_box_change(
    dummy_data_manager,
    search_box: str,
    first_index,
    first_text,
    shape,
    expect_result: bool,
):
    dummy_data_manager.search_value = search_box

    if expect_result:
        assert dummy_data_manager.table.items[0]["id"] == first_index
        assert dummy_data_manager.table.items[0]["text"] == f"<span>{first_text}</span>"
        assert dummy_data_manager.filtered_df["text"].shape == shape
    else:
        assert len(dummy_data_manager.table.items) == 0
        assert dummy_data_manager.filtered_df["text"].shape == shape


def test_labeling_raises_event_and_applies(dummy_data_manager):
    """Clicking on label buttons should raise an event and call our custom function,
    as well as modify the underlying data to include the new label"""
    returns = []

    def catch_label(index, new_label):
        nonlocal returns
        returns.append(index)
        returns.append(new_label)

    dummy_data_manager.on_data_labeled(catch_label)
    dummy_data_manager.apply_label(0, 1)
    assert returns[0] == 0
    assert returns[1] == 1
    assert dummy_data_manager.active_data.loc[0, dummy_data_manager.label_col] == 1


def test_clicking_non_label_col_raises_select_event(dummy_data_manager):
    """Clicking a non-button column should call the user's on row selected event"""
    select_returns = []

    def catch_select(index):
        nonlocal select_returns
        select_returns.append(index)

    dummy_data_manager.on_row_selected(catch_select)
    dummy_data_manager.fire_on_row_selected(2)
    assert select_returns[0] == 2


def test_clicking_label_on_filtered_view_still_works(dummy_data_manager):
    """Clicking a button when the view is a filtered DF should still work."""
    dummy_data_manager.sample_indices = [5]
    dummy_data_manager.data_tabs.value = "Sample"

    returns = []

    def catch_label(index, new_label):
        nonlocal returns
        returns.append(index)
        returns.append(new_label)

    dummy_data_manager.on_data_labeled(catch_label)
    dummy_data_manager.apply_label(5, 1)
    assert returns[0] == 5
    assert returns[1] == 1


@pytest.mark.integration
def test_apply_multiple_labels_sets_model_training_data(fun_df):
    """Calling apply_label with multiple indices/labels should correctly create initial
    multiple rows to the model's training data."""

    model = Model(fun_df, text_col="text")
    model.data.apply_label([0, 1, 2], [1, 1, 0])

    assert len(model.training_data) == 3


@pytest.mark.integration
def test_apply_multiple_labels_updates_model_training_data(fun_df):
    """Calling apply_label with multiple indices/labels should correctly update
    multiple rows to the model's training data even if those rows all already exist."""

    model = Model(fun_df, text_col="text")
    model.data.apply_label([0, 1, 2], [1, 1, 0])
    model.data.apply_label([0, 1, 2], [0, 0, 1])

    assert len(model.training_data) == 3
    assert model.training_data["_label"].values.tolist() == [0, 0, 1]


@pytest.mark.integration
def test_apply_multiple_labels_appends_model_training_data(fun_df):
    """Calling apply_label with multiple indices/labels should correctly append
    multiple new rows to the model's training data."""

    model = Model(fun_df, text_col="text")
    model.data.apply_label([0, 1, 2], [1, 1, 0])
    model.data.apply_label([3, 4, 5], [0, 0, 1])

    assert len(model.training_data) == 6
    assert model.training_data["_label"].values.tolist() == [1, 1, 0, 0, 0, 1]


def test_datamanager_with_none_text_col_doesnot_error(fun_df):
    """Technically it's allowed to have a data manager without a text col, it just
    shouldn't show anything."""
    DataManager(fun_df)


def test_datamanager_with_none_data_doesnot_error():
    """Technically it's allowed to have a data manager without any actual data, it just
    shouldn't show anything."""
    DataManager(None)


def test_setting_sample_indices_fires_event(dummy_data_manager):
    """Manually setting the sample of the data manager should fire the on_sample_changed event."""
    returns = []

    def catch_change(sample_indices):
        nonlocal returns
        returns.append(sample_indices)

    dummy_data_manager.sample_indices = [1, 2, 3, 4, 5, 6]
    dummy_data_manager.on_sample_changed(catch_change)
    dummy_data_manager.sample_indices = [1, 2, 3]
    assert len(returns) == 1
    assert returns[0] == [1, 2, 3]


@pytest.mark.integration
def test_adding_single_point_to_sample_fires_event(dummy_data_manager):
    """A sample added button click (from the data manager's event handler) should fire the
    on_sample_changed event."""
    returns = []

    def catch_change(sample_indices):
        nonlocal returns
        returns.append(sample_indices)

    dummy_data_manager.sample_indices = [1, 2, 3]
    dummy_data_manager.on_sample_changed(catch_change)
    dummy_data_manager.table.vue_addToSample(4)
    assert len(returns) == 1
    assert returns[0] == [1, 2, 3, 4]


def test_resampling_fires_event(dummy_data_manager):
    """Clicking the resample button should fire the on_sample_changed event."""
    returns = []

    def catch_change(sample_indices):
        nonlocal returns
        returns.append(sample_indices)

    dummy_data_manager.sample_indices = [1, 2, 3]
    dummy_data_manager.on_sample_changed(catch_change)
    dummy_data_manager.set_random_sample()
    assert len(returns) == 1
    assert returns[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_id_search_shows_only_id_row(dummy_data_manager):
    """Using a search string like ID:5 should only return a row with that index."""
    dummy_data_manager.search_value = "ID:5"
    assert len(dummy_data_manager.filtered_df) == 1
    assert (
        dummy_data_manager.filtered_df.iloc[0].text == "Hey kid, you can't skate here!"
    )
    assert dummy_data_manager.filtered_df.iloc[[0]].index[0] == 5


def test_id_search_correct_id_in_subsample(dummy_data_manager):
    """When a currently filtered dataframe is a subsample, the search ID should still
    be the returned ID."""
    dummy_data_manager.sample_indices = [4, 5, 6]
    assert len(dummy_data_manager.filtered_df) == 3
    dummy_data_manager.search_value = "ID:5"
    assert len(dummy_data_manager.filtered_df) == 1
    assert (
        dummy_data_manager.filtered_df.iloc[0].text == "Hey kid, you can't skate here!"
    )
    assert dummy_data_manager.filtered_df.iloc[[0]].index[0] == 5


def test_instance_click_in_non_sequential(non_sequential_df):
    """Clicking on a row with a non-sequential-id should still open the correct id
    in the instance viewer."""

    data = DataManager(non_sequential_df, text_col="text")
    data.table.vue_selectPoint(3)
    assert data.item_viewer.index == 3
