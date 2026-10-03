# model-promotion-bridge

[![ci](https://github.com/MicheleCampi/model-promotion-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/MicheleCampi/model-promotion-bridge/actions/workflows/ci.yml)

Reads what a model registry has promoted and emits the Kubernetes manifest that
serves exactly that revision. MLflow in, `VllmService` out, nothing in between.

## The gap this closes

A GPU session on 2026-07-04 left a [provenance file](https://github.com/MicheleCampi/vllm-coldstart-operator/blob/17cfcbb11aca9813d5110aad1b89aae6baa1949b/hack/gpu-session/runs/2026-07-04/PROVENANCE-session.txt)
that recorded, next to the driver and cluster versions, these two entries:

    == image digest
    vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
    == model snapshot
    a09a35458c702b33eeacc393d103063234e8bc28

The container image was pinned by digest. The model snapshot was a commit SHA
added by hand — the [session checklist](https://github.com/MicheleCampi/vllm-coldstart-operator/blob/17cfcbb11aca9813d5110aad1b89aae6baa1949b/hack/gpu-session/CHECKLIST.md#L72)
asked for the image digest, not for the model — because the manifest that
produced it declared only `model: Qwen/Qwen2.5-7B-Instruct`, a name and not a
revision. Two artefacts in the same pod, one reproducible and one not, and
nothing in the manifest showed the difference.

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
as [`--revision`](https://github.com/vllm-project/vllm/blob/84bcbc62644356270aaaa5e2d0237d03adc9bb3a/vllm/engine/arg_utils.py#L950)
and it takes effect, and in the annotations, where a reader who does not know
vLLM's flags can still see what was served and who decided it. vLLM applies the
same revision to the tokenizer unless `--tokenizer-revision` is given
([`vllm/config/model.py:590-591`](https://github.com/vllm-project/vllm/blob/84bcbc62644356270aaaa5e2d0237d03adc9bb3a/vllm/config/model.py#L590-L591)),
so `build_manifest` refuses either flag in the arguments a caller adds: a second
`--revision` would make the pin depend on argument order, and a
`--tokenizer-revision` would serve the pinned weights with a tokenizer from
elsewhere.

| Exit code | Meaning |
|---|---|
| 0 | manifest written |
| 3 | the registry answered and there is nothing to serve: no such alias or model, the reserved alias `latest`, or a promoted version without a revision tag |
| 4 | the registry could not answer |

1 and 2 are left to Python and argparse, so no outcome shares an exit code with
a crash. A pipeline can stop quietly on 3; 4 is an outage.

## Decisions worth arguing about

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

**An outage is never reported as "nothing promoted".** MLflow answers a missing
alias with `INVALID_PARAMETER_VALUE` and a missing model with
`RESOURCE_DOES_NOT_EXIST`; a refused connection surfaces as an `MlflowException`
with no code of its own, which defaults to `INTERNAL_ERROR` (mlflow 3.15.1:
`store/model_registry/sqlalchemy_store.py:1591-1593` and `:439-442`,
`utils/rest_utils.py:317`, `exceptions.py:79`). Only those two codes mean
"nothing to serve"; any other, including one this code has never seen, is an
outage. `RegistryUnavailable` is deliberately not a subclass of
`PromotionError`, so code written to stop quietly on one does not swallow the
other.

**`latest` is refused.** MLflow resolves the alias `latest`, in any letter
case, to the newest registered version (mlflow 3.15.1:
`utils/validation.py:57`, `store/model_registry/sqlalchemy_store.py:1571`):
the newest registration instead of the promoted one, which is the state this
exists to prevent.

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

The seed registers two versions carrying two real commits of the model's
Hugging Face repository — the one served on 2026-07-04 first, an earlier commit
second — and promotes the first. The emitted manifest serves that one: it
follows the alias, not the most recent registration, which is the property the
whole thing rests on.

## Tests and CI

    bash ci/install-tools.sh && bash ci/validate.sh

Where a test needs a registry, it gets a real MLflow registry on sqlite, and
where it needs one that cannot answer, MLflow's real HTTP client pointed at a
local port with nothing listening — not a mock. Mocking the client would have
tested my idea of the API instead of the API: MLflow 3.15.1 raises on the
filesystem store unless told otherwise (`store/tracking/file_store.py:224-234`),
which a mock would have hidden until the first real run. Each behavioural fix
landed after a commit adding the test that fails without it.

The same script runs on every push to `main` and every pull request:

- the tests, with every Python dependency installed from
  `requirements-dev.lock`, pinned to a version and a hash;
- the manifest the CLI emits from the seeded registry: it must carry the
  promoted revision, and it is validated with `kubeconform -strict` against the
  `VllmService` CRD of
  [vllm-coldstart-operator](https://github.com/MicheleCampi/vllm-coldstart-operator)
  at two releases — v0.2.1, the one
  [gke-llm-inference-platform](https://github.com/MicheleCampi/gke-llm-inference-platform)
  deploys, and v0.3.0 — each fetched at a fixed commit and checked by SHA-256.
  Field names in a generator are the kind of thing that looks right and is not;
- the CLI's exit codes, as a real process;
- gitleaks on the full history and actionlint on the workflow;
- a self-test: an unknown field and a revision that is not a string must each
  be rejected by both schemas, and a planted token by gitleaks, each for the
  reason named. A check that cannot fail proves nothing.
