.PHONY: test docker

test:
	coverage run --source=graph_rl/ -m pytest tests/
	coverage report -m

setup:
	poetry lock --no-update
	dephell deps convert
	black -l 100 setup.py

requirements:
	poetry export -f requirements.txt --output docker/requirements.txt
	poetry export -f requirements.txt --dev --output docker/dev_requirements.txt

reqs: setup requirements

black:
	black -l 100 .

docker:
	cd docker && \
	docker build -t graph_rl:test .
