#!/usr/bin/env bash
set -euo pipefail

RECIPE_ID="llama-cpp/qwen38-flash-next-ud-iq4-xs-qsa"

cat >&2 <<EOF_MSG
ERROR: ${RECIPE_ID} is an experimental kernel config, not the serving default.

The public launch path is recipes/llama-cpp/qwen38-flash-next-ud-iq4-xs/run.sh
(unpatched llama.cpp, ~25 tok/s short-prompt, 5.60 tok/s at 229k).

This directory only ships:
  patches/qsa-lightning-working.patch
  results/qsa-kernels.md  (locked hashes 2689367b205c16ce / 8547299278d81f66)

It refuses to start inference so the patched 64k path cannot replace the
unpatched recipe. Apply the patch on a dedicated Spark tree, rebuild, and
measure there. Do not mark this recipe verified.
EOF_MSG

exit 1
