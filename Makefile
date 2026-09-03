PYTHON ?= python

.PHONY: install bootstrap validate execute check
install:
	$(PYTHON) -m pip install -r artificial-neural-networks/requirements-notebook.txt

bootstrap:
	$(PYTHON) scripts/bootstrap_ann_notebooks.py

validate:
	$(PYTHON) scripts/validate_notebooks.py

execute:
	$(PYTHON) scripts/execute_notebooks.py --in-place

check: validate execute
	$(PYTHON) scripts/validate_notebooks.py --require-executed
