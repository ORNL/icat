import os
import shutil
import subprocess
import time

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


@pytest.fixture
def non_sequential_df(fun_df):
    return fun_df.drop([0, 1])


@pytest.fixture
def data_file_loc():
    shutil.rmtree("test/exampledata", ignore_errors=True)
    os.makedirs("test/exampledata", exist_ok=True)
    yield "test/exampledata/thing"
    shutil.rmtree("test/exampledata", ignore_errors=True)


@pytest.fixture
def jupyter_server():
    # os.system("jupyter lab --port 9997 --NotebookApp.token='' --NotebookApp.disable_check_xsrf='True'")
    process = subprocess.Popen(
        "jupyter lab ./notebooks --port 9997 --NotebookApp.token='' --NotebookApp.disable_check_xsrf='True' --no-browser",
        shell=True,
    )
    # process = subprocess.Popen("jupyter lab --port 9997 --NotebookApp.token=''", shell=True)
    time.sleep(2)
    yield
    process.kill()
    os.system("jupyter lab stop 9997")
    # app = jupyterlab_server.app.LabServerApp(settings={
    #     "ServerApp": {
    #         "token": "",
    #         "port": 9988,
    #     },
    # })
    # app.ma
    # yield
    # app.stop()

    # ServerApp.root_dir =
    # ServerApp.token = ""
