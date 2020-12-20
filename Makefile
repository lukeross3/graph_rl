test:
	coverage run --source=graph_rl/ -m pytest tests/
	coverage report -m