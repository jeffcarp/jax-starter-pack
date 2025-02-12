RUFF = ruff
FILES_TO_LINT = .

lint:
	$(RUFF) check $(FILES_TO_LINT)
