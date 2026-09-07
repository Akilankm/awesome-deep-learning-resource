install:
	pip install -r requirements.txt

bootstrap:
	python scripts/bootstrap_notebooks.py

execute: bootstrap
	python scripts/execute_notebooks.py

validate:
	python scripts/validate_notebooks.py
