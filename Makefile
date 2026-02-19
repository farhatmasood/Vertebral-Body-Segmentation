.PHONY: install test clean lint run

install:
	pip install .

install-dev:
	pip install -e .[dev]

test:
	pytest tests/

lint:
	mypy pymd/
	flake8 pymd/

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -delete

run:
	python -m pymd.cli
