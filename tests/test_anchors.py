import pandas as pd
import pytest

from icat.anchors import Anchor, DictionaryAnchor, TFIDFAnchor
from icat.model import Model


@pytest.mark.parametrize(
    "keywords,expected_count",
    [
        (["teapot"], 1.0),
        (["here"], 2.0),
        (["Here"], 2.0),
        (["short", "stout"], 2.0),
        (["idonotexist"], 0.0),
        (["idonotexist", "short"], 1.0),
        ([], 0.0),
        (["\n"], 0.0),
        ([""], 0.0),
    ],
)
def test_dictionary_anchor_keyword_count(
    dummy_string: str, keywords: list[str], expected_count: float
):
    """Keyword counting should work for all sorts of use cases."""
    test_anchor = DictionaryAnchor(text_col="", keywords=keywords, anchor_name="test")
    count = test_anchor._keyword_count(dummy_string)
    assert count == expected_count


def test_dictionary_anchor_featurize(fun_df: pd.DataFrame):
    """Featurize on the dictionary anchor should return the appropriate column."""
    test_anchor = DictionaryAnchor(
        text_col="text", keywords=["muffin", "I'm"], anchor_name="test"
    )
    new_series = test_anchor.featurize(fun_df)

    assert new_series.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        2.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_weighted_dictionary_anchor_keyword_count():
    """The output of a keyword count on a dictionary anchor should be multiplied by
    the anchor weight."""
    anchor = DictionaryAnchor(text_col="", keywords=["this"])
    text = "the word 'this' appears twice in this sentence."
    assert anchor._keyword_count(text) == 2.0
    anchor.weight = 2.0
    assert anchor._keyword_count(text) == 4.0


def test_weighted_dictionary_anchor_featurize(fun_df: pd.DataFrame):
    """Featurizing on a dictionary anchor that's weighted should return weighted values."""
    test_anchor = DictionaryAnchor(
        text_col="text", keywords=["muffin", "I'm"], anchor_name="test", weight=2.0
    )
    new_series = test_anchor.featurize(fun_df)

    assert new_series.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        4.0,
        0.0,
        4.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.panel
def test_dictionary_keyword_str_edits(dummy_anchor):
    """Editing a dictionary anchor's keywords_str should update the keywords list."""
    edit_string = "edit,to,string,test"
    edit_string_result = ["edit", "to", "string", "test"]

    dummy_anchor.keywords_str = edit_string
    assert dummy_anchor.keywords == edit_string_result


@pytest.mark.vue
def test_dictionary_keyword_input_edits(dummy_anchor):
    """Editing the keyword_inputs widget value should propagate to keywords_str and keywords."""
    edit_string = "edit,to,string,test"
    edit_string_result = ["edit", "to", "string", "test"]

    dummy_anchor._keywords_input.v_model = edit_string
    dummy_anchor._keywords_input.fire_event("change", edit_string)
    assert dummy_anchor.keywords_str == edit_string
    assert dummy_anchor.keywords == edit_string_result


@pytest.mark.panel
def test_dictionary_keyword_edits(dummy_anchor):
    """Editing a dictionary anchor's keywords should update the keywords_str."""
    edit_string_result = "edit,to,string,test"
    edit_string_list = ["edit", "to", "string", "test"]

    dummy_anchor.keywords = edit_string_list
    assert dummy_anchor.keywords_str == edit_string_result


@pytest.mark.vue
def test_dictionary_keyword_edits_keyword_inputs(dummy_anchor):
    """Editing the keywords value should update keyword_inputs value"""
    edit_string_result = "edit,to,string,test"
    edit_string_list = ["edit", "to", "string", "test"]

    dummy_anchor.keywords = edit_string_list
    assert dummy_anchor._keywords_input.v_model == edit_string_result


@pytest.mark.vue
def test_dictionary_keyword_str_edits_keyword_inputs(dummy_anchor):
    """Editing the keywords_str value should update keyword_inputs value"""
    edit_string_result = "edit,to,string,test"

    dummy_anchor.keywords_str = edit_string_result
    assert dummy_anchor._keywords_input.v_model == edit_string_result


@pytest.mark.vue
def test_anchor_init_starts_with_right_vmodels():
    """Initializing an anchor with set parameters should correctly populate the ipyvuetify inputs."""
    anchor = Anchor(anchor_name="trains", weight=2.1, in_view=False, in_model=False)

    assert anchor._anchor_name_input.v_model == "trains"
    assert anchor._weight_input.v_model == 2.1
    assert not anchor._in_view_input.v_model
    assert not anchor._in_model_input.v_model


@pytest.mark.vue
def test_anchor_kw_field_events_change_param():
    """Both the blur and change events for the keywords field should modify the param."""
    anchor = DictionaryAnchor(keywords=["beans", "and", "toast"])
    anchor._keywords_input.v_model = "beans,and"
    anchor._keywords_input.fire_event("change", "beans,and")
    assert anchor.keywords == ["beans", "and"]

    # blur simulates that we clicked away without hitting enter
    anchor._keywords_input.v_model = "beans"
    anchor._keywords_input.fire_event("blur", "beans")
    assert anchor.keywords == ["beans"]


@pytest.mark.panel
def test_dictionary_init_only():
    """Creating a dictionary passing "keywords" should also assign to keywords_str."""
    anchor = DictionaryAnchor(keywords=["beans", "and", "toast"])
    assert anchor.keywords_str == "beans,and,toast"


@pytest.mark.vue
def test_dictionary_init_inits_keyword_inputs():
    """Creating a dictionary anchor should populate the keyword_inputs with the passed keyword text."""
    anchor = DictionaryAnchor(keywords=["beans", "and", "toast"])
    assert anchor._keywords_input.v_model == "beans,and,toast"


@pytest.mark.vue
def test_anchor_name_changes_anchor_name_input():
    """Changing an anchor's anchor_name parameter should change the anchor_name_input value."""
    anchor = Anchor()
    anchor.anchor_name = "testing"

    assert anchor._anchor_name_input.v_model == "testing"


@pytest.mark.vue
def test_anchor_name_input_changes_anchor_name():
    """Changing an anchor's anchor_name_input should change the anchor_name parameter."""
    anchor = Anchor()
    anchor._anchor_name_input.v_model = "testing"
    anchor._anchor_name_input.fire_event("input", "testing")

    assert anchor.anchor_name == "testing"


@pytest.mark.vue
def test_weight_changes_weight_input():
    """Changing an anchor's weight parameter should change the weight_input value."""
    anchor = Anchor()
    anchor.weight = 3.5

    assert anchor._weight_input.v_model == 3.5


@pytest.mark.vue
def test_weight_input_changes_weight():
    """Changing an anchor's weight_input should change the weight parameter."""
    anchor = Anchor()
    anchor._weight_input.v_model = 3.5
    anchor._weight_input.fire_event("change", 3.5)

    assert anchor.weight == 3.5


@pytest.mark.vue
def test_in_view_changes_in_view_input():
    """Changing an anchor's in_view parameter should change the in_view_input value."""
    anchor = Anchor()
    anchor.in_view = False

    assert not anchor._in_view_input.v_model


@pytest.mark.vue
def test_in_view_input_changes_in_view():
    """Changing an anchor's in_view_input should change the in_view parameter."""
    anchor = Anchor()
    anchor._in_view_input.v_model = False
    anchor._in_view_input.fire_event("change", False)

    assert not anchor.in_view


@pytest.mark.vue
def test_in_model_changes_in_model_input():
    """Changing an anchor's in_model parameter should change the in_model_input value."""
    anchor = Anchor()
    anchor.in_model = False

    assert not anchor._in_model_input.v_model


@pytest.mark.vue
def test_in_model_input_changes_in_model():
    """Changing an anchor's in_model_input should change the in_model parameter."""
    anchor = Anchor()
    anchor._in_model_input.v_model = False
    anchor._in_model_input.fire_event("change", False)

    assert not anchor.in_model


def test_tfidf_anchor(fun_df):
    """Featurizing with a TFIDF anchor should result in the same text to be
    similarity 1, and ones containing a couple of same words should be nonzero."""
    anchor = TFIDFAnchor(text_col="text")
    anchor.reference_texts = [fun_df.loc[0, "text"]]
    results = anchor.featurize(fun_df)

    assert len(results) == 12
    assert results[0] == 1.0
    assert results[2] > 0.0
    assert results[4] > 0.0


def test_tfidf_change_texts_changes_short():
    """Setting the reference texts to a bunch of texts should set the 'shorts' to
    the shortened versions of those."""
    anchor = TFIDFAnchor(text_col="text")
    anchor.reference_texts = [
        "thing1",
        "This is a very long text that needs to be shorter",
    ]

    assert anchor.reference_short == ["thing1", "This is a very long text "]


@pytest.mark.integration
def test_tfidf_finds_id_from_text(fun_df):
    """Adding a text from a df to a TFIDF anchor should correctly find the row id."""
    model = Model(fun_df, "text")
    anchor = TFIDFAnchor(text_col="text")
    model.anchor_list.add_anchor(anchor)
    anchor.reference_texts = ["I like trains"]

    assert anchor.reference_short == ["1"]


def test_tfidf_anchor_with_multiple_texts(fun_df):
    """A tfidf anchor with multiple reference texts should average their vectors and
    use that for the similarity measurement."""
    anchor = TFIDFAnchor(text_col="text")
    anchor.reference_texts = [fun_df.loc[0, "text"], fun_df.loc[8, "text"]]
    results = anchor.featurize(fun_df)

    assert results[0] > 0.5
    assert results[2] > 0.0
    assert results[4] > 0.0

    assert results[8] > 0.5
    assert results[9] > 0.0


@pytest.mark.integration
def test_similarity_anchor_gets_short_from_text_entry(fun_df):
    """Adding raw text in the id_text field of a similarity anchor should correctly
    add to the reference_short and add a new chip"""

    model = Model(fun_df, "text")
    anchor = TFIDFAnchor(text_col="text")
    model.anchor_list.add_anchor(anchor)

    anchor._id_text.v_model = "kid"
    anchor._add_button.fire_event("click")

    assert anchor.reference_texts == ["kid"]
    assert anchor.reference_short == ["kid"]
    assert len(anchor._chips_container.children) == 1


def test_save_load_dictionary_anchor(data_file_loc):
    """Saving an anchor and then loading should be populated with all of the
    correct parameters."""

    a1 = DictionaryAnchor()
    a1.anchor_name = "I am an anchor"
    a1.weight = 1.2
    a1.in_view = False
    a1.in_model = False
    a1.text_col = "my_text"
    a1.keywords_str = "thing"
    a1.keywords = ["thing"]
    a1.cache["test"] = 13
    a1.save(data_file_loc)

    a2 = DictionaryAnchor()
    a2.load(data_file_loc)
    assert a2.anchor_name == "I am an anchor"
    assert a2.weight == 1.2
    assert not a2.in_view
    assert not a2.in_model
    assert a2.text_col == "my_text"
    assert a2.keywords_str == "thing"
    assert a2.keywords == ["thing"]
    assert a2.cache["test"] == 13
