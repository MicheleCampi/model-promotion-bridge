#!/usr/bin/env bash
# Validation, run identically in CI and locally:
#   bash ci/install-tools.sh && bash ci/validate.sh
# No cluster, no credentials, no network beyond the pinned downloads.
#
# Part 1 checks the bridge: its tests, the manifest it emits from a seeded
# registry, that manifest against the VllmService CRD of two operator
# releases, the CLI's exit codes as a real process, the history for secrets,
# the workflows. Part 2 is a self-test: each negative case must be rejected,
# and for the reason named, or the run fails. A check that cannot fail proves
# nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
CI_BIN=${CI_BIN:-$ROOT/.ci-bin}
export PATH="$CI_BIN:$PATH"
export PYTHONDONTWRITEBYTECODE=1
PY="$CI_BIN/venv/bin/python"
# v0.2.1 is the release gke-llm-inference-platform deploys; v0.3.0 the latest.
OPERATOR_RELEASES="v0.2.1 v0.3.0"
SERVED_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
. ci/platform-ci.env
# The helpers (pass, check, expect_reject, same, exit_code) come from
# inference-platform-ci, installed by ci/install-tools.sh. Without them every
# check would be undefined and fail would stay 0, so a missing lib.sh stops
# the run here instead of letting it report success.
. "$CI_BIN/platform-ci/lib.sh" || { echo "no $CI_BIN/platform-ci/lib.sh: run bash ci/install-tools.sh first" >&2; exit 1; }

echo "== shared CI files at the commit pinned in ci/platform-ci.env"
same platform-ci-commit "$(git -C "$CI_BIN/platform-ci" rev-parse HEAD)" "$PLATFORM_CI_COMMIT"

echo "== tests"
check pytest "$PY" -m pytest tests -q -p no:cacheprovider

echo "== manifest from a seeded registry (examples/seed_registry.py)"
REG="sqlite:///$W/registry.db"
check seed "$PY" examples/seed_registry.py --tracking-uri "$REG"
check render "$PY" -m bridge.cli --tracking-uri "$REG" --model qwen2.5-7b-instruct \
  --service-name qwen-prod --out "$W/qwen-prod.yaml"
# The seed registers the served revision first and an earlier one last, and
# promotes the first: the manifest must follow the alias, not the last write.
same served-revision "$("$PY" -c 'import sys, yaml; a = yaml.safe_load(open(sys.argv[1]))["spec"]["extraArgs"]; print(a[a.index("--revision") + 1])' "$W/qwen-prod.yaml")" "$SERVED_REVISION"

echo "== manifest against the VllmService CRD of each operator release (served versions only)"
cat > "$W/served.py" <<'PY'
import sys, yaml
out = []
for d in yaml.safe_load_all(sys.stdin):
    if d and d.get("kind") == "CustomResourceDefinition" and d["metadata"]["name"] in sys.argv[1:]:
        d["spec"]["versions"] = [v for v in d["spec"]["versions"] if v.get("served")]
        out.append(d)
assert len(out) == len(sys.argv[1:]), f"found {len(out)} of {len(sys.argv[1:])} CRDs"
yaml.safe_dump_all(out, sys.stdout)
PY
for r in $OPERATOR_RELEASES; do
  mkdir -p "$W/schemas/$r"
  check "schema-operator-$r" sh -c "'$PY' '$W/served.py' vllmservices.inference.michelecampi.dev < '$CI_BIN/crd-operator-$r.yaml' > '$W/crd-$r.yaml' && cd '$W/schemas/$r' && '$PY' '$CI_BIN/openapi2jsonschema.py' '$W/crd-$r.yaml'"
  check "manifest-operator-$r" kubeconform -strict -summary \
    -schema-location "$W/schemas/$r/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json" "$W/qwen-prod.yaml"
done

echo "== CLI exit codes, as a real process"
CLI=("$PY" -m bridge.cli --model qwen2.5-7b-instruct --service-name x)
exit_code cli-usage-error 2 "$PY" -m bridge.cli
exit_code cli-no-such-alias 3 "${CLI[@]}" --tracking-uri "$REG" --alias staging
exit_code cli-reserved-alias 3 "${CLI[@]}" --tracking-uri "$REG" --alias latest
exit_code cli-registry-unreachable 4 env MLFLOW_HTTP_REQUEST_MAX_RETRIES=0 \
  MLFLOW_HTTP_REQUEST_TIMEOUT=3 "${CLI[@]}" --tracking-uri http://127.0.0.1:9

echo "== secrets in the full history"
check gitleaks gitleaks git --redact --no-banner .

echo "== workflows"
check actionlint actionlint

echo "== self-test: each case must be rejected for the reason named"
mutate() { "$PY" -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); exec(sys.argv[3]); yaml.safe_dump(d, open(sys.argv[2], "w"))' "$W/qwen-prod.yaml" "$@"; }
mutate "$W/neg-unknown.yaml" 'd["spec"]["gpuz"] = 1'
mutate "$W/neg-number.yaml" 'd["spec"]["extraArgs"][1] = 8192'
for r in $OPERATOR_RELEASES; do
  KC=(kubeconform -strict -schema-location "$W/schemas/$r/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json")
  expect_reject "neg-unknown-field-$r" "additional properties 'gpuz' not allowed" "${KC[@]}" "$W/neg-unknown.yaml"
  expect_reject "neg-revision-not-a-string-$r" "extraArgs/1': got number, want string" "${KC[@]}" "$W/neg-number.yaml"
done
mkdir -p "$W/leak" && printf 'token = "ghp_%s"\n' "$(head -c 300 /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 36)" > "$W/leak/config.txt"
expect_reject neg-secret "leaks found: 1" gitleaks dir --redact --no-banner "$W/leak"

echo; if [ "$fail" = 0 ]; then echo "RESULT: all checks passed"; else echo "RESULT: failed"; fi
exit "$fail"
