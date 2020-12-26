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

install: reqs
	poetry install

black:
	black -l 100 .

test:
	coverage run --source=graph_rl/ -m pytest tests/
	coverage report -m

docker:
	cd docker && \
	docker build -t graph_rl:test .

all: reqs black test docker