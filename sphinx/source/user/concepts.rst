Concepts
########

ICAT is an example of an interactive machine learning (IML) tool - a tool where
a model is trained based on interactions with a human in the loop.

Specifically, ICAT provides interactive featuring and interactive labelling.


Underlying model
================

The primary object in icat is a ``Model``, which trains on the fly based on
the actions taken by the user. By default, this is a simple logistic regression
model provided by sklearn. A simpler model is required for this particular
use case, as it needs to be able to train within a few seconds, and since all
labels and features are provided on the fly by the user, the overall quantity
of training data and the complexity of the feature space is very low.

The model is assumed to be a binary classifier, with the intent that the user
indicates what is "interesting" verus "uninteresting". In practice, models can
be combined to achieve multi class predictions, or models can be used in sequence
to allow creating a chain of filters.

An ICAT model can be initialized by passing the dataset as a pandas dataframe
and the name of the column with the text to feature on:

.. code-block:: python

    import icat
    icat.initialize()

    model = icat.Model(my_data_df, "text_col")

Anchors
=======

Interactive featuring in ICAT is achieved through "anchors", terminology adopted
from the AnchorViz paper (`"AnchorViz: Facilitating classifier error discovery through interactive
semantic data exploration" <https://dl.acm.org/doi/abs/10.1145/3172944.3172950>`_.)

An anchor, conceptually, is an arbitrary function that returns a "strength" or
"attraction" value usually between 0 and 1. Simple examples include a
"dictionary" or bag of words that return the number of times some set of
keywords appear in each text, or a cosine-similarity score to a target text's
TF-IDF vector.

In implementation, an anchor in ICAT is any instance of a subclass of
``icat.anchors.Anchor``. All anchors have a ``featurize()`` function that runs
the actual feature computation, and is passed the data to featurize on. The
class implementation also has the ability to create a set of UI components to
configure it, which show up in the :ref:`Anchor list`

ICAT comes with several pre-defined anchors, (the ``DictionaryAnchor`` and
``TFIDFAnchor`` as defined above)

Anchors can be added in the interface by clicking on the associated anchor
type button in the anchorlist, or by programmatic definition:

.. code-block:: python

    import icat
    icat.initialize()
    model = icat.Model(my_data_df, "text_col")

    some_anchor = icat.DictionaryAnchor(anchor_name="news", keywords=["news"])
    model.add_anchor(some_anchor)

Labelling
=========

In the :ref:`Data manager`, the user has the ability to label any instance
as "interesting" or "uninteresting", with the "I" and "U" buttons respectively.
Every time the user supplies a new label, it is added to the underlying model's
training set.

Without an initial set of labels, the model is considered to be "unseeded", or
not having enough information to sufficiently train and make predictions. Once
the user supplies an inital set of 10 labeled instances, the model will train
and predict on the full dataset (coloring different parts in the interface to
reflect this - orange indicates "interesting" and blue indicates "uninteresting".)

Once a model has been seeded, all labelling and anchor modifications the user
makes retrain the model from scratch and updates the corresponding predictions.

Labelling can be done either with the available buttons in the data manager/item
viewer, or programmatically:

.. code-block:: python

    import icat
    icat.initialize()
    model = icat.Model(my_data_df, "text_col")

    model.data.apply_label(42, 1)  # label index 42 as "interesting"
    model.data.apply_label(13, 0)  # label index 13 as "uninteresting"

Selection
=========

One important interaction paradigm in ICAT is manually pulling out clusters of
interest in the data for further exploration or labelling. The anchors mentioned
above are the mechanism to pull data out from the center/away from other
anchors, and the interface additionally has the ability to lasso-select groups
of points by clicking and dragging a path around the target points:

.. figure:: ../_static/selection.png
   :align: center

Once points are selected/highlighted in green, the "Selected" tab of the :ref:`Data
manager` is populated with only the texts from these points, which can then be
labeled or analyzed for additional anchors/necessary anchor modifications.
