Getting Started
###############

Installation
============

The ICAT library itself can be installed via ``pip`` with:

.. code-block:: bash

    pip install icat-iml

ICAT is primarily intended to run in Jupyter, so it is recommended you
install and use either Jupyter Lab or Jupyter Notebook


Running
=======

To run ICAT, you have to first call ``initialize()``:

.. code-block:: python

    import icat
    icat.initialize()

To render the interface, create an icat ``Model`` with a pandas
dataframe with the data you want to explore and the name of the
column containing the text, and then execute ``.view`` at the end
of a cell:

.. code-block:: python

    model = icat.Model(my_data_df, "text_col")
    model.view


Simplest Example
================

Running the following two cells in a jupyter environment should
create a functioning interface:

.. code-block:: python

    import icat
    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups

    icat.initialize()

.. code-block:: python

    train = fetch_20newsgroups(subset="train")
    train_df = pd.DataFrame({"text": train["data"]})
    model = icat.Model(train_df, "text")
    model.view
