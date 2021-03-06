.PHONY: test docker

version:
	python3 scripts/update_version.py

setup:
	poetry lock --no-update
	dephell deps convert
	black -l 100 setup.py

requirements:
	poetry export -f requirements.txt --without-hashes --output docker/requirements.txt
	poetry export -f requirements.txt --without-hashes --dev --output docker/dev_requirements.txt

reqs: setup requirements version

install: reqs
	# poetry build
	# python3 -m pip install .
	# rm -rf dist/
	# rm -rf graph_rl.egg-info/
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

run-jupyter:
	jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --core-mode

all: install black test-mpi docker