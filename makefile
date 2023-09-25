# https://madewithml.com/courses/mlops/makefile
SHELL = /bin/bash
VERSION := $(shell python -c "import icat; print(icat.__version__)")

.PHONY: help
help:
	@echo "Commands:"
	@echo "pre-commit 	: run all pre-commit checks."
	@echo "apply-docs 	: copy current sphinx documentation into version-specific docs/ folder"
	@echo "style 		: executes style formatting."
	@echo "clean 		: cleans all unnecessary files."
	@echo "test 		: runs unit tests."

.PHONY: pre-commit
pre-commit:
	@pre-commit run --all-files

.PHONY: apply-docs
apply-docs:
	-@unlink docs/stable
	@echo "Copying documentation to docs/$(VERSION)"
	-@rm -rf docs/$(VERSION)
	@cp -r sphinx/build/html docs/$(VERSION)
	@echo "Linking to docs/stable"
	@ln -s $(VERSION)/ docs/stable


.PHONY: publish
publish:
	@python -m build
	@twine check dist/*
	@twine upload dist/* --skip-existing

.PHONY: style
style:
	black .
	flake8
	isort .

.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: test
test:
	pytest
