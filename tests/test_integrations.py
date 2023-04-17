# import pandas as pd
# import pandas as pd
import panel as pn

# from icat.data import DataManager
from icat.model import Model

# from sklearn.datasets import fetch_20newsgroups


def test_model_view_data(fun_df, dummy_anchor):
    pn.extension()
    model = Model(data=fun_df, text_col="text")
    model.anchor_list.add_anchor(dummy_anchor)
    model.view.refresh_data()
