"""Tests for the table of anchors."""

import pytest

from icat.anchorlist import AnchorList
from icat.anchors import Anchor, DictionaryAnchor, TFIDFAnchor
from icat.model import Model


def test_add_dictionary_button():
    """Clicking the add dictionary anchor button should, in fact, add a dictionary anchor."""
    al = AnchorList()

    returns = []

    def catch_addition(anchor):
        nonlocal returns
        returns.append(anchor)

    al.on_anchor_added(catch_addition)
    al._handle_ipv_new_anchor_generic_click(None, None, None, DictionaryAnchor)
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

    model2 = Model(fun_df, "text", anchor_types=[])
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

    assert len(model2.anchor_list.possible_anchor_types) == 2
    assert model2.anchor_list.default_example_anchor_type_dict["ref"] == TFIDFAnchor


# --- below may all go in another test file other
@pytest.mark.integration
def test_add_example_anchor_type_populates_example_type_dropdown():
    """Adding an anchor type that can be used for examples should add it to the
    example anchor type dropdown, and also auto assign if it hasn't already."""
    model = Model(None, "text", anchor_types=[])
    assert len(model.anchor_list.example_anchor_types_dropdown.items) == 0

    model.anchor_list.add_anchor_type(DictionaryAnchor)
    assert len(model.anchor_list.example_anchor_types_dropdown.items) == 0
    assert "ref" not in model.anchor_list.default_example_anchor_type_dict

    model.anchor_list.add_anchor_type(TFIDFAnchor)
    assert len(model.anchor_list.example_anchor_types_dropdown.items) == 1
    assert "ref" in model.anchor_list.default_example_anchor_type_dict
    assert model.data.table.example_type_name == "TF-IDF"
    assert model.data.table.example_btn_color == "#777777"


@pytest.mark.integration
def test_change_example_anchor_color_changes_av_and_table(fun_df):
    """Changing the color of the example anchor type should change both anchors in anchorviz
    and the button color in the table."""
    model = Model(fun_df, "text")
    anchor = TFIDFAnchor(anchor_name="test", reference_texts=["0"])
    model.add_anchor(anchor)

    model.anchor_list.modify_anchor_type(TFIDFAnchor, "color", "#00FF00")
    assert model.view.anchorviz.anchors[0]["color"] == "#00FF00"
    assert model.data.table.example_btn_color == "#00FF00"


def test_adding_anchor_type_adds_button():
    """When an anchor type is added to the anchorlist, a corresponding button should be
    added to main anchorlist section."""
    al = AnchorList(None, anchor_types=[])
    assert len(al.anchor_buttons.children) == 1

    al.add_anchor_type(DictionaryAnchor)
    assert len(al.anchor_buttons.children) == 3
    assert (
        al.anchor_buttons.children[2].v_slots[0]["children"].children[0] == "Dictionary"
    )


@pytest.mark.integration
def test_anchor_type_button_click_adds_anchor_of_type(fun_df):
    """The anchor buttons should add anchors of their appropriate types"""
    model = Model(fun_df, "text", anchor_types=[])
    assert len(model.anchor_list.anchor_buttons.children) == 1
    model.anchor_list.add_anchor_type(DictionaryAnchor, color="#FF0000")
    model.anchor_list.add_anchor_type(TFIDFAnchor, color="#0000FF")
    assert len(model.anchor_list.anchor_buttons.children) == 4

    assert (
        model.anchor_list.anchor_buttons.children[2].v_slots[0]["children"].children[0]
        == "Dictionary"
    )
    assert (
        model.anchor_list.anchor_buttons.children[2].v_slots[0]["children"].color
        == "#FF0000"
    )
    assert (
        model.anchor_list.anchor_buttons.children[3].v_slots[0]["children"].children[0]
        == "TF-IDF"
    )
    assert (
        model.anchor_list.anchor_buttons.children[3].v_slots[0]["children"].color
        == "#0000FF"
    )

    model.anchor_list.anchor_buttons.children[2].v_slots[0]["children"].fire_event(
        "click", None
    )
    assert type(model.anchor_list.anchors[0]) == DictionaryAnchor
    assert model.view.anchorviz.anchors[0]["color"] == "#FF0000"

    model.anchor_list.anchor_buttons.children[3].v_slots[0]["children"].fire_event(
        "click", None
    )
    assert type(model.anchor_list.anchors[1]) == TFIDFAnchor
    assert model.view.anchorviz.anchors[1]["color"] == "#0000FF"


def test_changing_anchor_type_color_changes_btn_color():
    """Setting the anchor type color should set the color of the associated button as well"""
    al = AnchorList(None, anchor_types=[{"ref": DictionaryAnchor, "color": "#FF00FF"}])
    assert len(al.anchor_buttons.children) == 3
    assert al.anchor_buttons.children[2].v_slots[0]["children"].color == "#FF00FF"

    al.modify_anchor_type(DictionaryAnchor, "color", "#330044")
    assert al.anchor_buttons.children[2].v_slots[0]["children"].color == "#330044"


def test_changing_anchor_type_name_changes_btn_text():
    """Setting the anchor type color should set the color of the associated button as well"""
    al = AnchorList(None, anchor_types=[{"ref": DictionaryAnchor}])
    assert len(al.anchor_buttons.children) == 3
    assert (
        al.anchor_buttons.children[2].v_slots[0]["children"].children[0] == "Dictionary"
    )

    al.modify_anchor_type(DictionaryAnchor, "name", "dict")
    assert al.anchor_buttons.children[2].v_slots[0]["children"].children[0] == "dict"


def test_defining_anchor_type_in_scope_adds_new_add_btn():
    """Defining a new anchor type should add a section for it in the anchor types tab."""

    class MyAnchor(Anchor):
        pass

    al = AnchorList()
    assert (
        al.anchor_types_layout.children[-1].children[0].children[0].v_model
        == "test_defining_anchor_type_in_scope_adds_new_add_btn.<locals>.MyAnchor"
    )


def test_defining_anchor_type_and_changin_name_adds_appropriately_named_btn():
    """When you add an anchor type it should use the name the user specified."""

    class MyAnchor(Anchor):
        pass

    al = AnchorList()
    al.anchor_types_layout.children[-1].children[0].children[0].v_model = "Something"
    al.anchor_types_layout.children[-1].children[-1].fire_event("click", None)
    assert (
        al.anchor_buttons.children[4].v_slots[0]["children"].children[0] == "Something"
    )


@pytest.mark.integration
def test_removing_default_example_anchor_type_resets(fun_df):
    """Removing the anchor type that is the default anchor type should reset
    the dropdown, the default_example_type property and the table."""
    model = Model(fun_df, "text")
    model.anchor_list.remove_anchor_type(TFIDFAnchor)
    assert len(model.anchor_list.example_anchor_types_dropdown.items) == 0
    assert model.anchor_list.default_example_anchor_type_dict == {}
    assert model.data.table.example_type_name == "similarity"
    assert model.data.table.example_btn_color == ""


@pytest.mark.integration
def test_removing_anchor_type_removes_those_anchors(fun_df):
    """Removing an anchor type should remove all associated anchors of that
    type, both in anchorviz and the list."""
    model = Model(fun_df, "text")
    a1 = DictionaryAnchor(keywords=["thing1"])
    a2 = DictionaryAnchor(keywords=["thing2"])
    model.add_anchor(a1)
    model.add_anchor(a2)
    assert len(model.anchor_list.anchors) == 2
    assert len(model.view.anchorviz.anchors) == 2
    model.anchor_list.remove_anchor_type(DictionaryAnchor)
    assert len(model.anchor_list.anchors) == 0
    assert len(model.view.anchorviz.anchors) == 0


def test_removing_anchor_type_removes_btn():
    """Removing an anchor type should remove the associated add button."""
    al = AnchorList(None, anchor_types=[])
    assert len(al.anchor_buttons.children) == 1

    al.add_anchor_type(DictionaryAnchor)
    assert len(al.anchor_buttons.children) == 3
    assert (
        al.anchor_buttons.children[2].v_slots[0]["children"].children[0] == "Dictionary"
    )

    al.remove_anchor_type(DictionaryAnchor)
    assert len(al.anchor_buttons.children) == 2


def test_loading_anchorlist_doesnot_add_duplicate_anchor_types(fun_df, data_file_loc):
    """Saving an anchorlist and loading it in a new anchorlist that had default types
    should not end up with duplicate types and buttons."""
    al1 = AnchorList(None)
    al1.save(data_file_loc)

    al2 = AnchorList(None)
    al2.load(data_file_loc)
    assert len(al2.possible_anchor_types) == 2
    assert len(al2.anchor_buttons.children) == 4  # 4 because toggle button and p "new"
