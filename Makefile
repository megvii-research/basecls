format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r basecls test zoo
	isort basecls test zoo
	black basecls test zoo
	flake8 basecls test zoo

lint:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r basecls test zoo
	isort --diff --check basecls test zoo
	black --diff --check --color basecls test zoo
	flake8 basecls test zoo

unittest:
	pytest --cov=basecls test
	coverage xml
