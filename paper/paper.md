---
title: 'ICAT: The Interactive Corpus Analysis Tool'
tags:
  - Python
  - Machine Learning
  - HCI
  - Visual Analytics
authors:
  - name: Nathan Martindale
    corresponding: true
    orcid: 0000-0002-5036-5433
    affiliation: 1
  - name: Scott L. Stewart
    orcid: 0000-0003-4320-5818
    affiliation: 1
affiliations:
 - name: Oak Ridge National Laboratory
   index: 1
date: 1 November 2023
bibliography: paper.bib
---

# Summary

The Interactive Corpus Analysis Tool (ICAT) is a Python library for creating dashboards to explore textual datasets and build simple binary classification models to help filter through them and focus on entries of interest. This tool uses a form of interactive machine learning (IML), a paradigm of "machine teaching" [@simardMachineTeachingNew2017] that sits at the intersection of the fields of human computer interaction (HCI), visual analytics, and machine learning. The intent of ICAT is to allow subject matter experts (SME) with limited to no experience in machine learning to benefit from an iterative human-in-the-loop (HITL) approach to building their own model without needing to understand the details of the underlying algorithm. This interactivity is achieved by allowing the user to create features, label data points, and visually manipulate a representation of the features to manually cluster and investigate data, while a model is trained on the fly based on these actions. ICAT is built on top of the Panel [@panelholoviz] library, using a combination of Vega, a custom IPyWidget using D3, and ipyvuetify, and is intended to be used inside of a Jupyter environment.

# Statement of Need

Machine teaching promises to democratize machine learning algorithms and grant non-machine-learning experts the ability to train, manipulate, and work with models themselves [@simardMachineTeachingNew2017]. Traditionally, the process for an SME to obtain a model that aids in their data analysis is a time consuming iterative loop: they must first communicate their problem space and data to a machine learning expert, who experiments and trains a model for the SME, who then tests it and finds any issues or insufficiently learned concepts, which must then be communicated back to the ML expert, and the iterative loop continues as such. Ideally, an effective HITL training process involves the SME more directly in the training process, dramatically speeding up this iteration loop and benefiting from the SME's implicit knowledge and experience. IML seeks to provide this process through mechanisms such as feature selection (interactive featuring) and model steering (interactive labeling) [@dudleyReviewUserInterface2018].

This is a challenging space for a number of reasons. The efficacy of an IML system heavily revolves around the design of the interface itself, in addition to the underlying machine learning models and the many considerations they entail. Thus, incorporating effective user experience design principles and understanding the mental models of the users as they explore and use the interface is crucial. Both quantitative and qualitative metrics must include the human element, so any research seeking to demonstrate a measured value-add or efficacy of an IML interface must incorporate user studies [@laiScienceHumanAIDecision2023]. A positive user experience additionally constrains algorithmic design in terms of speed and efficiency--an underlying model that takes minutes to train is frustrating to interact with [@failsInteractiveMachineLearning]. Care must be taken not to treat the user like a mechanical turk or mindless oracle for the model to endlessly query [@amershiPowerPeopleRole2014],[@cakmakDesigningInteractionsRobot2010].

Despite these challenges, there is tremendous potential for IML to empower SMEs and allow them to benefit from the value of machine learning in their work. For the field to grow and realize this potential, a great deal more research and work are required. Our work draws heavily on the IML interface concepts proposed by Suh and colleagues in 2019 [@suhAnchorVizFacilitatingSemantic2019], and as of this writing there is no other open source package implementing their visuals or overall interface. ICAT seeks to fill this gap to allow other researchers to explore, build on, and compare against the concepts discussed below to further the state of the field.

# Interface Concepts

The ICAT workflow trains a simple binary classification model to separate "uninteresting" from "interesting" entries in an initially unlabeled dataset. "Interesting" here is intentionally vague to refer to whatever classification signal a user is implicitly or explicitly attempting to extract. The primary use case for this is for ICAT to help filter a large collection of text objects to some smaller target subset of interest that is easier to manually review.

ICAT implements both interactive featuring and interactive labeling. Interactive featuring allows the user to create or influence the feature columns that the underlying model uses for training and predicting. In ICAT this is done through "concept anchors," or functions that return some value, nominally between 0 and 1, to represent a strength or "pull" on a provided input. An example anchor type included with ICAT is a dictionary or keyword anchor, where the feature value is a bag of words or count of the number of occurences of each keyword the user provides to that anchor. Interactive labeling is done by allowing the user to manually specify or change a label (uninteresting or interesting) for any text entry. All labeled rows are used as the training dataset for the underlying model. Once the model is "seeded," or given some minimum number of labels to train on, it begins to predict on the full dataset, and the visualizations in the dashboard are colored to reflect the interesting/uninteresting prediction for each entry.

Anchors are visualized with the AnchorViz ring [@suhAnchorVizFacilitatingSemantic2019], which are shown below in \autoref{fig:dashboard}(a). The visual representation of anchors are draggable points around the circumference of the ring, with the entries from a sample of the data rendered as the smaller points inside. Anchors pull points toward them based on the strength of influence or the magnitude of the feature value on each point. As the user drags the anchors around, the associated points then similarly move according to these varying attraction strengths, and this helps visually determine the overlap between anchors and which points are or are not represented well by the current feature set. This ability to manually position the anchors and their correpsonding points allows for various strategies for manually clustering the data, such as using anchors to pull away any distractors or known incorrect keywords from the interesting set, and so on.

![An example of a rendered dashboard from ICAT. Throughout the dashboard, blue points and text indicate uninteresting, orange indicate interesting. (a) The AnchorViz ring. (b) The data manager, explorer, and labeling tool. (c) The anchor list/feature editing section.\label{fig:dashboard}](icat_labeled.png)

The dashboard also includes a data manager, shown in \autoref{fig:dashboard}(b), with tables containing various subsets of the data to make it easier to scroll through and explore the text. The data tabs have five filters that can be applied to the data. These include showing only the current sample of points in the AnchorViz display, showing all points that have been labeled, showing all points the model has predicted are interesting, or showing the result of the user lasso-selecting arbitrary points in the visualization. The tables also include a set of actions per row, allowing the user to apply labels to the points, create example anchors (by default adding a new ``TFIDFAnchor`` with the chosen row as the similarity target), and add them to the current sample if they are not already included.

Below the AnchorViz ring is the anchor list, shown in \autoref{fig:dashboard}(c). The anchor list contains all of the current anchors, statistics about the number of points they cover and their classification breakdown, and the associated controls for modifying their parameters. Every anchor type can have a customized set of controls to display when a corresponding anchor row is expanded in the anchor list. This can consist of any combination of ipyvuetify [@ipyvuetify] elements stored in the ``.widget`` instance of the anchor. An anchor type is effectively a wrapper around a function that computes a feature value, and ICAT can support dynamically adding new anchor types to the interface for any class inheriting from ``icat.anchors.Anchor``.

As discussed earlier, an important concept for a tool that trains a model on the fly based on user interaction is to respond and train quickly [@failsInteractiveMachineLearning]. _Interactive featuring_ means that a single user action could change the feature values for the entire training dataset, which requires retraining the underlying model from scratch. Since all features and labels are user provided, the overall volume and necessary complexity is relatively low, and by default ICAT uses scikit-learn's [@scikit-learn] logistic regression algorithm. This results in a training time of a few seconds on most modern laptop hardware with datasets smaller than 50,000 entries when using the default anchor types that come with ICAT.

# Acknowledgments

The authors would like to acknowledge the US Department of Energy, National Nuclear Security Administration's Office of Defense Nuclear Nonproliferation Research and Development (NA-22) for supporting this work.

This manuscript has been authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan ([http://energy.gov/downloads/doe-public-access-plan](http://energy.gov/downloads/doe-public-access-plan)).

# References
