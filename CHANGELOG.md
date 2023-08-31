# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Fixed
* Issue where calling load on `AnchorList` with default anchor types would potentially
    add duplicate anchor types.
* AnchorList limiting to 10 items per page.




## [0.7.0] - 2023-08-25

### Added
* Anchor types tab to anchorlist, anchor classes in scope are automatically detected,
    and can be dynamically added and removed.
* Color delineation to anchors, each row in the anchorlist and anchor in anchorviz is
    colored corresponding to anchor color (configurable in anchor types tab.)
* Ability to change what anchor type is used when the "example" buttons are clicked.
* Optional description and name class attributes to anchors for display in the interface.
* Individual anchor-level `cache` dictionary that gets loaded and saved with the anchor
* Access to anchor list cache through an anchor `global_cache` property.

### Removed
* Similarity functions from model constructor and the similarity function anchor
    (subclass the `SimilarityAnchorBase` and provide it with a `featurize()`
    function instead - see the `lm_similarity_example` notebook.)




## [0.6.0] - 2023-08-22

### Added
* Offline mode arg to ``icat.initialize()``.
* Sample size configuration through model constructor and text field in sample tab.

### Changed
* Interesting and uninteresting colors to orange and blue respectively (swapped to better indicate
    'warm' as interesting and 'cold' as uninteresting.)
* "Instance" viewer to "item" viewer.

### Fixed
* Instance viewer missing search highlight.




## [0.5.0] - 2023-08-21

### Added
* Index searching to data manager.
* Search highlighting in data manager.
* "Unlabel" option to remove previously labeled data from training set.

### Fixed
* Data column sorting not working.
* Needing to render the notebook's first model view cell twice for ipyvuetify
    stylesheets to propagate. (Use `icat.initialize()` now, instead of
    `pn.extension()`)




## [0.4.0] - 2023-08-18

### Added
* "All" tab in data manager to allow searching through entire dataset.
* Sample buttons for data manager rows that aren't already in sample.
* Coverage stats for unseeded models.
* Buttons to add to anchors from search box.

### Fixed
* Issues adding two anchors with the same name.
* Data manager not updating highlighting after new anchor added.




## [0.3.0] - 2023-08-14

### Added
* Useful imports to top level module.
* Activity indicators to anchors in anchor list table.
* `add_anchor` function directly to model.
* Saving/loading functionality to anchors, anchor lists, and models.
* Add expand/collapse all anchors in anchor list button.
* Buttons for label all interesting/uninteresting to selected data tab.

### Fixed
* Text fields not applying changes when user clicks away instead of hitting enter.
* Similarity function anchor dropdown change not triggering featurization.
* Data manager's active data columns incorrectly updating with anchor names if model
    hasn't been trained yet.




## [0.2.0] - 2023-07-05

### Added
* Random initial anchor placement when added via buttons.
* Similarity function anchor, allowing user to provide arbitrary
    similarity functions to the model and use them for anchor featurization.

### Changed
* Panel version to >1.0.

### Fixed
* Data manager search box doing a cased rather than uncased search.
* Data manager unable to switch back to 'sample' tab.
* Data manager not updating highlighting after keyword anchor change
    until some other UI update.
* Too small default interface height.




## [0.1.2] - 2023-04-17

### Fixed
* More missing dependencies




## [0.1.1] - 2023-04-17

### Fixed
* Missing dependencies




## [0.1.0] - 2023-04-17

First open source release!
