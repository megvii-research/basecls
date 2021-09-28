format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r basecls doc test zoo
	isort basecls doc test zoo
	black basecls doc test zoo
	flake8 basecls doc test zoo

lint:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r basecls doc test zoo
	isort --diff --check basecls doc test zoo
	black --diff --check --color basecls doc test zoo
	flake8 basecls doc test zoo

test:
	pytest --cov=basecls test
	coverage xml
