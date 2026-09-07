install:
	pip install -r requirements.txt
	pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

bootstrap:
	python scripts/bootstrap_notebooks.py

execute: bootstrap
	python scripts/execute_notebooks.py

validate:
	python scripts/validate_notebooks.py
