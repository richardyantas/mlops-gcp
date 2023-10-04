conda-update:
	conda env update -f environment.yml
pip-tools:
	pip-compile requirements/dev.in && pip-compile requirements/prod.in
	pip-sync requirements/dev.txt requirements/prod.txt
setup:
	cp requirements/dev.txt ./pipeline
	cp requirements/prod.txt ./app 