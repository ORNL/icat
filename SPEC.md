## Function naming

* Events should all be named with past tense e.g. "anchor_removed"

* Functions that fire events/call relevant callbacks should be prefixed with `fire_on_`, with a name matching the callback adding function specified below.
    * Fire functions should _only_ run through the callbacks
* Functions that allow adding relevant callbacks should be prefixed with `on_`
* Functions that are the the receiver of events/actually handle them should be prefixed with `_handle_` (these functions should not directly be called by any code external to the class it appears in)
    * if the handler is explicitly for ipyvuetify, use `_handle_ipv` so it's obvious that it shouldn't be called even internally
    * if the handler is explicitly for ipywidgets, use `_handle_ipw`
    * if the handler is explicitly for panel, use `_handle_pnl`
