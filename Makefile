# Makefile

HOOKS_DIR = .git/hooks

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-docs

.PHONY: clean-build
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr deployments/build/
	rm -fr deployments/Dockerfiles/open_aea/packages
	rm -fr pip-wheel-metadata
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -name '*.svn' -exec rm -fr {} +
	find . -name '*.db' -exec rm -fr {} +
	rm -fr .idea .history
	rm -fr venv

.PHONY: clean-docs
clean-docs:
	rm -fr site/

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	rm -fr .tox/
	rm -f .coverage
	find . -name ".coverage*" -not -name ".coveragerc" -exec rm -fr "{}" \;
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .hypothesis
	rm -fr .pytest_cache
	rm -fr .mypy_cache/
	find . -name 'log.txt' -exec rm -fr {} +
	find . -name 'log.*.txt' -exec rm -fr {} +

.PHONY: hashes
hashes: clean
	poetry run autonomy packages lock
	poetry run autonomy push-all


.PHONY: poetry-install
poetry-install: 

	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry run pip install --upgrade --force-reinstall setuptools==75.9.1  # fix for KeyError: 'setuptools._distutils.compilers'



./agent:  poetry-install ./hash_id
	@if [ ! -d "agent" ]; then \
		poetry run autonomy -s fetch --remote `cat ./hash_id` --alias agent; \
	fi \


.PHONY: build-agent-runner
build-agent-runner: poetry-install agent
	poetry run pyinstaller \
	--collect-data eth_account \
	--collect-all aea \
	--collect-all autonomy \
	--collect-all aea_ledger_ethereum \
	--collect-all aea_ledger_cosmos \
	--hidden-import aea_ledger_ethereum \
	--hidden-import aea_ledger_cosmos \
	$(shell poetry run python get_pyinstaller_dependencies.py) \
	--onefile pyinstaller/mindshare_bin.py \
	--name agent_runner_bin
	./dist/agent_runner_bin --version
	

.PHONY: build-agent-runner-mac
build-agent-runner-mac: poetry-install  agent
	poetry run pyinstaller \
	--collect-data eth_account \
	--collect-all aea \
	--collect-all autonomy \
	--collect-all aea_ledger_ethereum \
	--collect-all aea_ledger_cosmos \
	--hidden-import aea_ledger_ethereum \
	--hidden-import aea_ledger_cosmos \
	$(shell poetry run python get_pyinstaller_dependencies.py) \
	--onefile pyinstaller/mindshare_bin.py \
	--name agent_runner_bin
	./dist/agent_runner_bin --version


./agent:  poetry-install ./hash_id
	@if [ ! -d "agent" ]; then \
		poetry run autonomy -s fetch --remote `cat ./hash_id` --alias agent; \
	fi \

./hash_id: ./packages/packages.json
	cat ./packages/packages.json | jq -r '.dev | to_entries[] | select(.key | startswith("agent/")) | .value' > ./hash_id

./agent_id: ./packages/packages.json
	cat ./packages/packages.json | jq -r '.dev | to_entries[] | select(.key | startswith("agent/")) | .key | sub("^agent/"; "")' > ./agent_id

./agent.zip: ./agent
	zip -r ./agent.zip ./agent

./agent.tar.gz: ./agent
	tar czf ./agent.tar.gz ./agent

./agent/ethereum_private_key.txt: ./agent
	poetry run bash -c "cd ./agent; autonomy  -s generate-key ethereum; autonomy  -s add-key ethereum ethereum_private_key.txt; autonomy -s issue-certificates;"

.PHONY: check-agent-runner
check-agent-runner:
	python check_agent_runner.py

lint:
	poetry run adev -v -n 0 lint

fmt: 
	poetry run adev -n 0 fmt

test:
	poetry run adev -v test

install:
	@echo "Setting up Git hooks..."

	# Create symlinks for pre-commit and pre-push hooks
	cp scripts/pre_commit_hook.sh $(HOOKS_DIR)/pre-commit
	cp scripts/pre_push_hook.sh $(HOOKS_DIR)/pre-push
	chmod +x $(HOOKS_DIR)/pre-commit
	chmod +x $(HOOKS_DIR)/pre-push
	@echo "Git hooks have been installed."
	@echo "Installing dependencies..."
	bash install.sh
	@echo "Dependencies installed."
	@echo "Syncing packages..."
	poetry run autonomy packages sync
	@echo "Packages synced."

 sync:
	git pull
	poetry run autonomy packages sync

all: fmt lint test hashes