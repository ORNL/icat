# https://madewithml.com/courses/mlops/makefile
SHELL = /bin/bash
VERSION := $(shell python -c "import icat; print(icat.__version__)")

.PHONY: help
help: ## display all the make commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: pre-commit
pre-commit: ## run all pre-commit checks
	@pre-commit run --all-files

.PHONY: apply-docs
apply-docs: ## copy current sphinx documentation into version-specific docs/ folder
	-@unlink docs/stable
	@echo "Copying documentation to docs/$(VERSION)"
	-@rm -rf docs/$(VERSION)
	@cp -r sphinx/build/html docs/$(VERSION)
	@echo "Linking to docs/stable"
	@ln -s $(VERSION)/ docs/stable


.PHONY: publish
publish: ## build and upload latest version to pypi
	@python -m build
	@twine check dist/*
	@twine upload dist/* --skip-existing

.PHONY: style
style: ## run autofixers and linters
	black .
	flake8
	isort .

.PHONY: clean
clean: ## remove auto generated cruft files
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: test
test: ## run unit tests
	pytest

.PHONY: test_debug
test_debug: ## run unit tests with playwright debugging enabled
	PWDEBUG=1 pytest -s
