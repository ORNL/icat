"""Utility functions that are needed in multiple places."""

import re


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


def populate_anchor_from_dictionary(anchor, parameters: dict[str, any]):
    """Assign anchor's parameters from a dictionary containing the key value pairs.

    This is useful to call from an anchor's ``load()`` implementation.

    Args:
        anchor (Anchor): The anchor to assign the parameters to.
        parameters (dict[str, any]): The parameters to assign to the anchor.
    """
    for key, value in parameters.items():
        setattr(anchor, key, value)


def add_highlights(text: str, regex: str, color: str = "yellow") -> str:
    """Adds HTML span tag highlights around any matches in the text for the given regex.

    Args:
        text (str): The text to add highlights to.
        regex (str): The regular expression to search for and sub in the text. Note that \
            this needs to have one capture group, so the regex should be wrapped in '()'.
        color (str): The background color to highlight the text with.

    Note:
        The regular expression will be treated as *case insensitive*.
    """
    # if blank regular expression, don't modify, including this here just
    # simplifies some of the places I'm calling this function.
    if regex == "" or regex == "()":
        return text

    return re.sub(
        regex,
        f"<span style='background-color: {color}; color: black;'>\\g<1></span>",
        text,
        flags=re.IGNORECASE,
    )
