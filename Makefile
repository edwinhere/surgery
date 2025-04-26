.PHONY: run watch install venv

# Default variance threshold if not provided
VARIANCE_THRESHOLD ?= 0.90

venv:
	uv venv
	# No need to activate here as we'll use the full path to binaries

install: venv
	uv pip install -r requirements.txt

run: install
	./.venv/bin/python main.py --variance-threshold $(VARIANCE_THRESHOLD)

watch: install
	@echo "Watching for changes in Python files..."
	@while true; do \
		find . -name "*.py" | entr -d -r ./.venv/bin/python main.py --variance-threshold $(VARIANCE_THRESHOLD); \
	done 