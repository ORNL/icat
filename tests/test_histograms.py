"""Tests for the histogram component and histograms column."""

import pytest

from icat.histogram import Histogram
from icat.model import Model


def test_compute_bins_correct(fun_df, dummy_anchor):
    """Running compute bins with some basic data should correctly
    compute a histogram."""
    hist = Histogram()

    model = Model(fun_df, text_col="text")
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

    hist._compute_bins(model.data.active_data, model.data.prediction_col)
    assert hist.data.shape == (50, 3)
    assert hist.data.loc[0.08, "count"] == 9
    assert hist.data.loc[0.88, "count"] == 3
    assert hist.data.loc[0.08, "bin_end"] == 0.10
    assert hist.data.loc[0.88, "bin_end"] == 0.90


@pytest.mark.integration
def test_labelling_data_updates_histograms(fun_df, dummy_anchor):
    """Labelling a new point should update the histograms through
    the changing predictions from the model."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    for i in range(11):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)

    local_bins = model.view.histograms.hist_local.data.copy()
    global_bins = model.view.histograms.hist_global.data.copy()

    model.data.apply_label(4, 0)

    local_bins_2 = model.view.histograms.hist_local.data.copy()
    global_bins_2 = model.view.histograms.hist_global.data.copy()

    local_is_different = False
    for index, row in local_bins.iterrows():
        if row.count != local_bins_2.loc[index, "count"]:
            local_is_different = True
            break

    global_is_different = False
    for index, row in global_bins.iterrows():
        if row.count != global_bins_2.loc[index, "count"]:
            global_is_different = True
            break

    assert local_is_different
    assert global_is_different


def test_local_vs_global_histogram_has_only_sample(fun_df, dummy_anchor):
    """The data passed to the local histogram should only consist of the sample
    data, while the global should be the full dataset."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    for i in range(10):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)
    model.data.sample_indices = [1, 2, 3]
    model.data.apply_label(10, 0)
    assert model.view.histograms.hist_local.data["count"].sum() == 3
    assert model.view.histograms.hist_global.data["count"].sum() == 12


@pytest.mark.vue
def test_slider_change_triggers_event(fun_df):
    """Changing the range of the slider should correctly fire an event handler
    and pass the range in a scale from 0-1."""

    returns = []

    def catch_change(range):
        nonlocal returns
        returns.append(range)

    model = Model(fun_df, text_col="text")
    model.view.histograms.on_range_changed(catch_change)
    model.view.histograms.slider.v_model = [30, 70]
    model.view.histograms.slider.fire_event("change", [30, 70])

    assert len(returns) == 1
    assert returns[0] == [0.3, 0.7]


@pytest.mark.integration
def test_reduced_range_reduces_av_and_data_sample(fun_df, dummy_anchor):
    """Changing the range should remove out-of-range entries from anchorviz and
    the data manager sample tab."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    for i in range(11):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)

    assert len(model.view.anchorviz.data) == 12
    assert model.data.filtered_df.shape[0] == 12

    model.view.histograms.slider.v_model = [50, 60]
    model.view.histograms.slider.fire_event("change", [50, 60])

    assert model.data.filtered_df.shape[0] < 12
    assert len(model.view.anchorviz.data) < 12
