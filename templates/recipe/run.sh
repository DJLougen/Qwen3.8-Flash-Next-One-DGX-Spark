#!/usr/bin/env bash
set -euo pipefail

RECIPE_ID="{{RECIPE_ID}}"
RUNTIME="{{RUNTIME}}"

cat >&2 <<EOF_MSG
ERROR: Recipe '${RECIPE_ID}' is a draft template.

Inference is not configured. This fail-closed draft guard refuses to execute until you
replace it with a real ${RUNTIME} invocation for the model specified in
recipe.json.

The workspace generator does not fabricate working inference commands. Edit
run.sh with the concrete server or batch command for your weights, then
validate the recipe before marking it verified.

See README.md in this directory for authoring steps.
EOF_MSG

exit 1
