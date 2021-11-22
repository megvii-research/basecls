format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r basecls test
	isort basecls test
	black basecls test
	flake8 basecls test

lint:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r basecls test
	isort --diff --check basecls test
	black --diff --check --color basecls test
	flake8 basecls test

unittest:
	pytest --cov=basecls test
	coverage xml
