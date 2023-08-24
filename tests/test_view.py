# NOTE: these are primarily integration tests

import pytest

from icat.anchors import DictionaryAnchor, TFIDFAnchor
from icat.model import Model


@pytest.mark.integration
def test_add_anchor_to_list_adds_to_viz(fun_df, dummy_anchor):
    """Adding an anchor to a model's anchorlist should add it to the viz."""
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    assert len(model.view.anchorviz.anchors) == 1
    assert len(model.anchor_list.anchors) == 1
    assert model.view.anchorviz.anchors[0]["id"] == dummy_anchor.name
    assert model.view.anchorviz.anchors[0]["name"] == dummy_anchor.anchor_name


@pytest.mark.integration
def test_new_viz_anchor_adds_to_list(fun_df):
    """A new anchor added in the viz should trigger appropriate events and add
    that anchor to the anchorlist."""
    model = Model(fun_df, text_col="text")
    anchor_data = dict(id="2", theta="-1.25146", name="New Anchor")
    model.view.anchorviz.anchors = [anchor_data]
    model.view.anchorviz._anchor_add_callbacks(
        {
            "type": "anchor_add",
            "theta": -1.2514631214807919,
            "newAnchor": anchor_data,
        }
    )
    assert len(model.anchor_list.anchors) == 1
    assert len(model.view.anchorviz.anchors) == 1


# def test_add_anchor_without_explicit_text_col_is_populated(fun_df):
#     """The anchorlist adding a dictionary anchor won't have a text_col by
#     default, this should be added by model and shouldn't crash."""
#     model = Model(fun_df, text_col="text")
#     model.anchor_list._on_add_dictionary_anchor()


# TODO: move this to model tests
def test_add_anchor_from_list_button_works(fun_df):
    """The anchorlist adding a dictionary anchor won't have a text_col by
    default, this should be added by model and shouldn't crash."""
    model = Model(fun_df, text_col="text")
    model.anchor_list._handle_ipv_new_anchor_generic_click(
        None, None, None, DictionaryAnchor
    )
    assert len(model.anchor_list.anchors) == 1
    assert model.anchor_list.anchors[0].text_col == "text"


@pytest.mark.integration
def test_triggering_selected_indices_updates_datamanger(fun_df):
    """Changing the set of lassoed points on anchorviz should flow
    through to the data manager's selected indices."""
    model = Model(fun_df, text_col="text")
    model.view.anchorviz.lassoedPointIDs = ["1", "2", "3"]
    # model.view._trigger_selected_points_change(None)
    assert model.data.selected_indices == [1, 2, 3]


@pytest.mark.integration
def test_hovering_row_selects_point_in_av(fun_df):
    """Hovering over a row in the datatable should change the selected
    point in the anchorviz."""
    model = Model(fun_df, text_col="text")
    model.data.table.vue_hoverPoint(5)
    assert model.view.anchorviz.selectedDataPointID == "5"


@pytest.mark.integration
def test_data_is_sent_to_av_normalized(fun_df):
    """The view should normalize rows with values > 1 before they are sent to anchorviz."""
    model = Model(fun_df, text_col="text")
    anchor = DictionaryAnchor(container=model.anchor_list, keywords=["beep"])
    model.anchor_list.add_anchor(anchor)
    points = model.view._serialize_data_to_dicts()
    assert points[4]["weights"][anchor.name] == 1.0


@pytest.mark.integration
def test_predicted_model_add_tfidf(fun_df, dummy_anchor):
    """Getting to a predictions state and adding a tf-idf anchor shouldn't crash."""
    fun_df
    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    for i in range(11):
        if i in [4, 6, 7]:
            model.data.apply_label(i, 1)
        else:
            model.data.apply_label(i, 0)

    tfidf_anchor = TFIDFAnchor(text_col="text")
    model.anchor_list.add_anchor(tfidf_anchor)
    tfidf_anchor.reference_texts = [fun_df.loc[0, "text"], fun_df.loc[1, "text"]]
    assert model.anchor_list.anchors[-1].reference_short == ["0", "1"]
