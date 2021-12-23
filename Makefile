format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r basecls docs test zoo
	isort basecls docs test zoo
	black basecls docs test zoo
	flake8 basecls docs test zoo

lint:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r basecls docs test zoo
	isort --diff --check basecls docs test zoo
	black --diff --check --color basecls docs test zoo
	flake8 basecls docs test zoo

unittest:
	pytest --cov=basecls test
	coverage xml
