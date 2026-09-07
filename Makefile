.PHONY: install build execute validate verify lab

install:
	python -m pip install -r requirements.txt
	python -m pip install tensorflow-cpu==2.18.0

build:
	python scripts/build_curriculum.py

execute:
	python scripts/execute_notebooks.py

validate:
	python scripts/validate_notebooks.py

verify: build execute validate

lab:
	jupyter lab
