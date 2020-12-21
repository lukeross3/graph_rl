test:
	coverage run --source=graph_rl/ -m pytest tests/
	coverage report -m

setup:
	dephell deps convert
	black setup.py

requirements:
	poetry lock
	poetry export --output docker/requirements.txt

reqs: setup requirements

docker-build:
	cd docker && \
	docker build -t graph_rl:test .
