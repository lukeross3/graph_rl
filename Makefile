.PHONY: test docker

version:
	python3 scripts/update_version.py

setup:
	poetry lock --no-update
	dephell deps convert
	black -l 100 setup.py

requirements:
	poetry export -f requirements.txt --output docker/requirements.txt
	poetry export -f requirements.txt --dev --output docker/dev_requirements.txt

reqs: setup requirements version

# TODO: try adding new module and see if this fixes weird import issues
install: reqs
	poetry build
	python3 -m pip install .
	rm -r dist/
	rm -r graph_rl.egg-info/
	poetry install

black:
	black -l 100 .

test:
	coverage run --source=graph_rl/ -m pytest tests/
	coverage report -m
	rm .coverage*

test-mpi:
	mpiexec -np 2 coverage run --source=graph_rl/ -p -m pytest tests/ --with-mpi
	coverage run --source=graph_rl/ --append -m pytest tests/
	coverage combine -a
	coverage report -m
	rm .coverage*

docker:
	cd docker && \
	docker build -t graph_rl:test .

all: install black test-mpi docker