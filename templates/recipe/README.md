# {{TITLE}}

> **Draft recipe template.** This directory was generated from the workspace authoring template.
> Inference is **not** configured and **will not run** until you replace the fail-closed
> draft guard in `run.sh` with a real {{RUNTIME}} invocation for the model below.

## Summary

{{SUMMARY}}

## Model

- Repository: {{MODEL_REPOSITORY}}
- Revision: {{MODEL_REVISION}}

## Runtime artifact provenance

If the runtime loads converted or quantized weights, document the source model revision separately from the output artifact. Record the conversion tool revision and command/config, quantization policy, output repository revision, shard/index names, and SHA-256 checksums. Mixed-bit MoE recipes must state which layers or experts use each bit width. Conversion completion alone is not inference verification.

## Hardware target

- Platform: NVIDIA DGX Spark
- GPU: GB10

## Authoring checklist

1. Copy variables from `env.example` into your shell or a local env file.
2. Replace the fail-closed draft guard in `run.sh` with the concrete {{RUNTIME}} server or inference command.
3. Exercise the recipe on DGX Spark hardware and capture any tuning notes here.
4. Run workspace validation before promoting `status` from `draft`.

## Identifiers

- Recipe ID: `{{RECIPE_ID}}`
- Runtime ID: `{{RUNTIME}}`
