#!/usr/bin/env bash
# Fetch inference-platform-ci at the tag pinned in ci/platform-ci.env and
# require the tag to resolve to the pinned commit before running anything
# from it (so this check cannot come from its own lib.sh). Then install into
# $CI_BIN, each SHA-256 verified, the shared tools the bridge uses and the
# operator CRDs of ci/tools.lock, and requirements-dev.lock into $CI_BIN/venv.
# Linux amd64 only. PLATFORM_CI_ENV replaces ci/platform-ci.env, for testing
# that a wrong pin fails.
set -euo pipefail
cd "$(dirname "$0")/.."
CI_BIN=${CI_BIN:-$PWD/.ci-bin}
. "${PLATFORM_CI_ENV:-ci/platform-ci.env}"
P=$CI_BIN/platform-ci
rm -rf "$P" && mkdir -p "$CI_BIN" && git init -q "$P"
git -C "$P" fetch -q --depth 1 https://github.com/MicheleCampi/inference-platform-ci.git "refs/tags/$PLATFORM_CI_TAG:refs/tags/$PLATFORM_CI_TAG"
got=$(git -C "$P" rev-parse --verify -q "refs/tags/$PLATFORM_CI_TAG^{commit}")
if [ "$got" != "$PLATFORM_CI_COMMIT" ]; then echo "inference-platform-ci $PLATFORM_CI_TAG resolves to $got, not the pinned $PLATFORM_CI_COMMIT" >&2; exit 1; fi
git -C "$P" checkout -q "$PLATFORM_CI_COMMIT"
bash "$P/install.sh" "$CI_BIN" "$P/tools.lock" kubeconform gitleaks actionlint openapi2jsonschema
bash "$P/install.sh" "$CI_BIN" ci/tools.lock
bash "$P/install-python.sh" "$CI_BIN/venv" requirements-dev.lock
