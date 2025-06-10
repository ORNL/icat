SHELL=/usr/bin/env bash
VERSION=$(shell python -c "import icat; print(icat.__version__)")
MM_INIT=eval "$$(micromamba shell hook --shell bash)"
MAMBA=micromamba
ENV_NAME=icat

.PHONY: help
help: ## display all the make commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## make a dev environment for this project from scratch (vars: MAMBA, ENV_NAME)
	-$(MAMBA) env remove -n $(ENV_NAME) -y
	$(MAMBA) env create -n $(ENV_NAME) -f environment.yml -y
	$(MAMBA) run -n $(ENV_NAME) pip install -r requirements.txt
	$(MAMBA) run -n $(ENV_NAME) playwright install
	$(MAMBA) run -n $(ENV_NAME) pre-commit install
	@echo -e "Environment created, activate with:\n\n$(MAMBA) activate $(ENV_NAME)"

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
	pytest -s

.PHONY: test_debug
test_debug: ## run unit tests with playwright debugging enabled
	PWDEBUG=1 pytest -s


.PHONY: scratch
scratch: ## build an environment completely from scratch and run tests
	-micromamba env remove -n icat-scratch -y
	micromamba create -n icat-scratch python -y
	$(MM_INIT) && micromamba activate icat-scratch && pip install -e . && pip install -r requirements.txt && playwright install && pytest -s

.PHONY: paper
paper: ## build the JOSS article
	docker run --rm \
		--volume ./paper:/data \
		--user $(id -u):$(id -g) \
		--env JOURNAL=joss \
		openjournals/inara
