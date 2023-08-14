"""Tests for the table of anchors."""

import pytest

from icat.anchorlist import AnchorList
from icat.anchors import DictionaryAnchor
from icat.model import Model


def test_add_dictionary_button():
    """Clicking the add dictionary anchor button should, in fact, add a dictionary anchor."""
    al = AnchorList()

    returns = []

    def catch_addition(anchor):
        nonlocal returns
        returns.append(anchor)

    al.on_anchor_added(catch_addition)
    al._handle_pnl_new_dictionary_btn_clicked(None)
    assert len(al.anchors) == 1
    assert len(al.table.items) == 1
    assert len(returns) == 1
    assert type(al.anchors[0]) == DictionaryAnchor
    assert returns[0] == al.anchors[0]


def test_anchor_change_fires_from_anchor_changes(dummy_anchor_list, dummy_anchor):
    """Modifying anchors should trigger the anchorlist event handler."""
    returns = []

    def catch_change(name, key, value):
        nonlocal returns
        returns.extend([name, key, value])

    # add another anchor to double check that making changes to more than
    # just the first row still works.
    dummy_anchor_list.add_anchor(DictionaryAnchor(anchor_name="test2"))
    dummy_anchor_list.on_anchor_changed(catch_change)

    dummy_anchor._in_view_input.v_model = False
    dummy_anchor._in_view_input.fire_event("change", False)

    assert len(returns) == 3
    assert returns[0] == dummy_anchor_list.anchors[0].name
    assert returns[1] == "in_view"
    assert returns[2] is False


def test_anchor_manual_change_triggers_event(dummy_anchor_list):
    """Programatically updating an anchor should trigger an event."""
    returns = []

    def catch_change(name, key, value):
        nonlocal returns
        returns.extend([name, key, value])

    immutable_name = dummy_anchor_list.anchors[0].name
    dummy_anchor_list.on_anchor_changed(catch_change)

    dummy_anchor_list.anchors[0].anchor_name = "hello!"
    assert returns[0] == immutable_name
    assert returns[1] == "anchor_name"
    assert returns[2] == "hello!"


def test_dictionary_keyword_edits_triggers_event(dummy_anchor_list, dummy_anchor):
    """Programmatically setting both keywords_str and keywords on a dictionary anchor
    should trigger an event."""
    returns = []

    def catch_change(name, key, value):
        nonlocal returns
        returns.extend([name, key, value])

    dummy_anchor_list.on_anchor_changed(catch_change)

    edit_string = "edit,to,string,test"
    edit_string_array = ["edit", "test"]

    dummy_anchor.keywords_str = edit_string
    assert len(returns) == 3
    assert returns[2] == ["edit", "to", "string", "test"]
    dummy_anchor.keywords = edit_string_array
    assert len(returns) == 6
    assert returns[5] == edit_string_array


def test_replacing_entire_anchor_list_correctly_injects_container(dummy_anchor_list):
    """The self injection of the anchorlist into the anchor's container should
    run when we replace the anchors list."""
    new_anchor = DictionaryAnchor(anchor_name="things")
    dummy_anchor_list.anchors = [new_anchor]
    assert new_anchor.container == dummy_anchor_list


def test_deleting_anchor_removes_from_table(dummy_anchor_list, dummy_anchor):
    """Using the anchorlist's remove anchor function should remove it from the table."""
    dummy_anchor_list.remove_anchor(dummy_anchor)
    assert len(dummy_anchor_list.table.items) == 0


def test_delete_button_removes_anchor_from_list(dummy_anchor_list, dummy_anchor):
    """The delete button on the anchorlist table should remove the corresponding anchor
    from the list."""
    dummy_anchor_list.table.vue_deleteAnchor(dummy_anchor.name)
    assert len(dummy_anchor_list.table.items) == 0
    assert len(dummy_anchor_list.anchors) == 0


@pytest.mark.integration
def test_delete_button_removes_anchor_from_anchorviz(fun_df, dummy_anchor):
    """The delete button on the anchorlist table should remove the corresponding anchor
    from anchorviz."""

    model = Model(fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    model.anchor_list.table.vue_deleteAnchor(dummy_anchor.name)

    assert len(model.view.anchorviz.anchors) == 0


@pytest.mark.parametrize(
    "kw_set1,kw_set2,expected_result",
    [
        (["test"], None, "(test)"),
        # a single anchor with one keyword should simply be the
        # keyword with parentheses.
        (["test1", "test2"], None, "(test1|test2)"),
        # a single anchor with multiple keywords should be parentheses
        # with the keywords split by the | "or" operator.
        (None, None, ""),
        # regex for no dictionary anchors should be a blank string.
        ([], None, ""),
        # a dictionary anchor with no keywords should be a blank string.
        (["test1", "test2"], [], "(test1|test2)"),
        # two dictionary anchors, one of which has no keywords, should
        # be the keywords of only the non-empty one with no extra '|'
        (["test1", "test2"], [""], "(test1|test2)"),
        # two dictionary anchors, one of which has an empty string keyword, should
        # be the keywords of only the non-empty-string one with no extra '|'
        (["test1", "test2"], ["test3", "test4"], "(test1|test2|test3|test4)"),
        # two non-empty dictionary anchors should have all keywords
        # flat joined with '|' surrounded by parentheses"""
        ([""], None, ""),
        # A dictionary with a keyword that is an empty string should
        # result in an empty string
        (["test1", ""], None, "(test1)"),
        # a dictionary with one keyword that is an empty string shouldn't
        # include the empty string with an extra '|' (blank strings have
        # the potential to match everything!)
        ([r"(test1*)"], None, r"(\(test1\*\))")
        # special characters need to be escaped, otherwise something like
        # a * could break all the keyword highlighting
    ],
)
def test_highlight_regex(kw_set1, kw_set2, expected_result):
    """Constructing a regex from dictionary anchor keywords should correctly follow
    the defined set of rules above."""
    anchor_list = AnchorList()
    if kw_set1 is not None:
        anchor_list.add_anchor(DictionaryAnchor(anchor_name="set1", keywords=kw_set1))
    if kw_set2 is not None:
        anchor_list.add_anchor(DictionaryAnchor(anchor_name="set2", keywords=kw_set2))

    assert anchor_list.highlight_regex() == expected_result

    """Special characters should be escaped, esp *"""


def test_save_load_anchorlist(data_file_loc, fun_df):
    """When we save an anchorlist and then reload it, all the anchors should reload into
    the same spot with the same parameters."""
    model = Model(fun_df, "text")
    a1 = DictionaryAnchor(anchor_name="thing1")
    a1.keywords = ["hello", "there"]
    a1.theta = 1.2
    model.anchor_list.cache["test_cache"] = 13

    a2 = DictionaryAnchor(anchor_name="thing2")
    a2.keywords = ["world", "here"]
    a2.theta = 1.5

    model.add_anchor(a1)
    model.add_anchor(a2)

    model.anchor_list.save(data_file_loc)

    model2 = Model(fun_df, "text")
    model2.anchor_list.load(data_file_loc)
    assert model2.anchor_list.cache["test_cache"] == 13

    a21 = model2.anchor_list.anchors[0]
    a22 = model2.anchor_list.anchors[1]
    assert a21.anchor_name == "thing1"
    assert a22.anchor_name == "thing2"
    assert a21.keywords == ["hello", "there"]
    assert a22.keywords == ["world", "here"]

    for anchor in model2.view.anchorviz.anchors:
        if anchor["id"] == a21.name:
            assert anchor["theta"] == 1.2
        elif anchor["id"] == a22.name:
            assert anchor["theta"] == 1.5
        else:
            raise Exception("what.")
