# model-promotion-bridge

Reads what a model registry has promoted and emits the Kubernetes manifest that
serves exactly that revision. MLflow in, `VllmService` out, nothing in between.

## The gap this closes

A GPU session on 2026-07-04 left a provenance file with two lines in it:

    == image digest
    vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
    == model snapshot
    a09a35458c702b33eeacc393d103063234e8bc28

The container image was pinned by digest. The model snapshot was a commit SHA
typed in by hand *after* the run, because the manifest that produced it declared
only `model: Qwen/Qwen2.5-7B-Instruct` — a name, not a revision. Two artefacts in
the same pod, one reproducible and one not, and nothing in the manifest showed
the difference.

This closes that. The registry owns which revision is current; this renders it
into a manifest; GitOps applies it; the operator serves it.

## What it does

    python -m bridge.cli \
      --tracking-uri sqlite:///mlflow.db \
      --model qwen2.5-7b-instruct \
      --alias production \
      --service-name qwen-prod \
      --out clusters/prod/qwen-prod.yaml

The revision lands in two places on purpose: in `extraArgs`, where vLLM reads it
and it takes effect, and in the annotations, where a reader who does not know
vLLM's flags can still see what was served and who decided it.

## Three decisions worth arguing about

**It emits YAML rather than calling the Kubernetes API.** A manifest is
reviewable before it lands, diffable against what is running, and replayable
from git a year later. An API call is none of those, and this way the bridge
needs no cluster credentials.

**The operator does not know MLflow exists.** Coupling a platform component to
one vendor's registry is a cost paid on every future change; a manifest is the
cheaper contract. The registry decides *what*, GitOps applies, the operator
serves — three concerns kept apart.

**A promoted version with no revision tag is an error, not a default.** Falling
back to `main` would emit a manifest that deploys whatever the model repo
happens to hold that day, which is the exact state this exists to prevent. The
pipeline stops instead.

## What this is not

It does not train, fine-tune or evaluate anything. The models are third-party
weights served as they are, and the registry records their provenance rather
than claiming authorship — the honest shape for a platform that serves models it
did not produce.

It is also not a promotion *policy*. Deciding that a version deserves the
`production` alias is a human or pipeline judgement; this reads the decision and
does not make it.

## Reproducing the example

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python examples/seed_registry.py --tracking-uri sqlite:///demo.db
    .venv/bin/python -m bridge.cli --tracking-uri sqlite:///demo.db \
      --model qwen2.5-7b-instruct --service-name qwen-prod

The seed registers two versions with different revisions and promotes the
**older** one. The emitted manifest serves that one — following the alias, not
the newest registration, which is the property the whole thing rests on.

## Tests

    .venv/bin/python -m pytest tests/ -q

Six tests against a real MLflow registry on sqlite rather than a mock: mocking
the client would have tested my idea of the API instead of the API, and MLflow
3.15 moved the filesystem store into maintenance mode in a way a mock would
have hidden until the first real run.

Four of the six cover refusals rather than the happy path, and each was checked
by breaking the thing it guards. Replacing the missing-revision error with a
fallback to `main` turns one of them red — which is how I know it is a test and
not decoration.

## Verified against the real CRD

The emitted manifest is validated with `kubectl apply --dry-run=server` against
the schema [vllm-coldstart-operator](https://github.com/MicheleCampi/vllm-coldstart-operator)
generates, on a kind cluster. Field names in a generator are the kind of thing
that looks right and is not.
