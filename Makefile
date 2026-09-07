install:
	pip install -r requirements.txt

execute:
	python scripts/execute_notebooks.py

validate:
	python scripts/validate_notebooks.py

verify: execute validate
