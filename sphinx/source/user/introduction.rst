Introduction
============

The Interactive Corpus Analysis Tool (ICAT) is a Python library for creating dashboards to explore textual datasets and build simple binary classification models to help filter through them and focus on entries of interest. This tool uses a form of interactive machine learning (IML), a paradigm of "machine teaching" [1]_ that sits at the intersection of the fields of human computer interaction (HCI), visual analytics, and machine learning. The intent of ICAT is to allow subject matter experts (SME) with limited to no experience in machine learning to benefit from an iterative human-in-the-loop (HITL) approach to building their own model without needing to understand the details of the underlying algorithm. This interactivity is achieved by allowing the user to create features, label data points, and visually manipulate a representation of the features to manually cluster and investigate data, while a model is trained on the fly based on these actions. ICAT is built on top of the Panel [2]_ library, using a combination of Vega, a custom IPyWidget using D3, and ipyvuetify, and is intended to be used inside of a Jupyter environment.

Statement of Need
-----------------


Machine teaching promises to democratize machine learning algorithms and grant non-machine-learning experts the ability to train, manipulate, and work with models themselves [1]_. Traditionally, the process for an SME to obtain a model that aids in their data analysis is a time consuming iterative loop: they must first communicate their problem space and data to a machine learning expert, who experiments and trains a model for the SME, who then tests it and finds any issues or insufficiently learned concepts, which must then be communicated back to the ML expert, and the iterative loop continues as such. Ideally, an effective HITL training process involves the SME more directly in the training process, dramatically speeding up this iteration loop and benefiting from the SME's implicit knowledge and experience. IML seeks to provide this process through mechanisms such as feature selection (interactive featuring) and model steering (interactive labeling) [3]_.

This is a challenging space for a number of reasons. The efficacy of an IML system heavily revolves around the design of the interface itself, in addition to the underlying machine learning models and the many considerations they entail. Thus, incorporating effective user experience design principles and understanding the mental models of the users as they explore and use the interface is crucial. Both quantitative and qualitative metrics must include the human element, so any research seeking to demonstrate a measured value-add or efficacy of an IML interface must incorporate user studies [4]_. A positive user experience additionally constrains algorithmic design in terms of speed and efficiency--an underlying model that takes minutes to train is frustrating to interact with [5]_. Care must be taken not to treat the user like a mechanical turk or mindless oracle for the model to endlessly query [6]_ [7]_.

Despite these challenges, there is tremendous potential for IML to empower SMEs and allow them to benefit from the value of machine learning in their work. For the field to grow and realize this potential, a great deal more research and work are required. Our work draws heavily on the IML interface concepts proposed by Suh and colleagues in 2019 [8]_, and as of this writing there is no other open source package implementing their visuals or overall interface. ICAT seeks to fill this gap to allow other researchers to explore, build on, and compare against the concepts discussed below to further the state of the field.




.. [1] Simard, P. Y., Amershi, S., Chickering, D. M., Pelton, A. E., Ghorashi, S., Meek, C., Ramos,
    G., Suh, J., Verwey, J., Wang, M., & Wernsing, J. (2017). Machine Teaching: A New
    Paradigm for Building Machine Learning Systems. http://arxiv.org/abs/1707.06742

.. [2] Holoviz. (2018). Panel: The powerful data exploration & web app framework for python. https://panel.holoviz.org/

.. [3] Dudley, J. J., & Kristensson, P. O. (2018). A Review of User Interface Design for Interactive Machine Learning. ACM Transactions on Interactive Intelligent Systems, 8(2), 8:1–8:37. https://doi.org/10.1145/3185517

.. [4] Lai, V., Chen, C., Smith-Renner, A., Liao, Q. V., & Tan, C. (2023). Towards a Science of Human-AI Decision Making: An Overview of Design Space in Empirical Human-Subject Studies. 2023 ACM Conference on Fairness, Accountability, and Transparency, 1369–1385. https://doi.org/10.1145/3593013.3594087

.. [5] Fails, J. A., & Olsen, D. R. (2003). Interactive machine learning. Proceedings of the 8th International Conference on Intelligent User Interfaces, 39–45. https://doi.org/10.1145/604045.604056

.. [6] Amershi, S., Cakmak, M., Knox, W. B., & Kulesza, T. (2014). Power to the People: The Role of Humans in Interactive Machine Learning. AI Magazine, 35(4, 4), 105–120. https://doi.org/10.1609/aimag.v35i42513

.. [7] Cakmak, M., Chao, C., & Thomaz, A. L. (2010). Designing Interactions for Robot Active Learners. IEEE Transactions on Autonomous Mental Development, 2(2), 108–118. https://doi.org/10.1109/TAMD.2010.2051030

.. [8] Suh, J., Ghorashi, S., Ramos, G., Chen, N.-C., Drucker, S., Verwey, J., & Simard, P. (2019). AnchorViz: Facilitating Semantic Data Exploration and Concept Discovery for Interactive Machine Learning. ACM Transactions on Interactive Intelligent Systems, 10(1), 7:1–7:38. https://doi.org/10.1145/3241379
