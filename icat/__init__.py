__version__ = "0.2.0"


def _kill_param_auto_docstring():
    """As cool as param.parameterized is, it's doing some things with automatic
    docstring generation that break my sphinx docs, and end up spitting a lot of
    incorrectly formatted (for RST docs) ANSI color code-ridden html. There's
    actually an example of this in params docs too:
    https://param.holoviz.org/reference.html#param.parameterized.default_label_formatter

    This is technically probably only an issue because sphinx autodoc actually
    imports the code in order to get the docstrings.

    To fix for modules that directly or indirectly inherit from
    param.parameterized.Parameterized, import and run this function
    at the top of the file.
    """
    import param.parameterized

    param.parameterized.docstring_describe_params = False
    param.parameterized.docstring_signature = False
