install:
	pip install -r requirements.txt
	pip install tensorflow-cpu==2.18.0

bootstrap:
	python scripts/bootstrap_notebooks.py

execute: bootstrap
	python scripts/execute_notebooks.py

validate:
	python scripts/validate_notebooks.py
