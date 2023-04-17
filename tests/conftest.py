import pandas as pd
import pytest

from icat.anchorlist import AnchorList
from icat.anchors import DictionaryAnchor
from icat.data import DataManager


@pytest.fixture
def dummy_anchor() -> DictionaryAnchor:
    return DictionaryAnchor(
        anchor_name="test", keywords=["I'm", "muffin"], text_col="text"
    )


@pytest.fixture
def dummy_anchor_list(dummy_anchor):
    al = AnchorList()
    al.add_anchor(dummy_anchor)
    return al


@pytest.fixture
def dummy_data_manager(fun_df):
    data = DataManager(data=fun_df, text_col="text")
    return data


@pytest.fixture
def dummy_string() -> str:
    return """I'm a little teapot
    Short and stout
    Here is my handle,
    Here is my spout.
    When I get all steamed up
    Hear me shout:
    Tip me over
    And pour me out!"""


@pytest.fixture
def fun_df():
    rows = [
        {"text": "They said I could never teach a llama to drive!"},
        {"text": "I like trains"},
        {"text": "No llama, no!"},
        {"text": "You are a chair, darling."},
        {"text": "Beep. Beep. I'm a sheep. I said beep beep I'm a sheep."},
        {"text": "Hey kid, you can't skate here!"},
        {"text": "Ow, hey, muffin man do you ever run out of muffins?"},
        {"text": "I'm going to punch your face. IN THE FACE."},
        {"text": "Oh boy a pie, what flavor?"},
        {"text": "PIE FLAVOR."},
        {"text": "Joey did you eat my sandwich?"},
        {"text": "I am your sandwich."},
    ]
    return pd.DataFrame(rows)
