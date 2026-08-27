.PHONY: new validate test check help

help:
	@echo "Targets:"
	@echo "  new      Interactively create a draft recipe"
	@echo "  validate Validate all recipe manifests"
	@echo "  test     Run repository unit tests"
	@echo "  check    Run validate + test"

new:
	python3 scripts/new_recipe.py --interactive

validate:
	python3 scripts/validate_recipes.py

test:
	python3 -m unittest discover -s tests -v

check: validate test
