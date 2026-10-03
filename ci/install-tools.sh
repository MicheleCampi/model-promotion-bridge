#!/usr/bin/env bash
# Download every artifact pinned in ci/tools.lock into $CI_BIN, verifying each
# one's SHA-256 before using it, then install the Python dependencies from the
# hash-locked requirements-dev.lock into $CI_BIN/venv. Linux amd64 only.
set -euo pipefail
cd "$(dirname "$0")/.."
CI_BIN=${CI_BIN:-$PWD/.ci-bin}
dl=$(mktemp -d); trap 'rm -rf "$dl"' EXIT
mkdir -p "$CI_BIN"
while read -r name url sha; do
  case "$name" in ''|'#'*) continue ;; esac
  # Prefixed with the name: both operator CRDs are called crd.yaml upstream.
  f="$dl/$name.$(basename "$url")"
  curl -fsSL --retry 3 -o "$f" "$url"
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then echo "checksum mismatch for $name: expected $sha, got $got" >&2; exit 1; fi
  case "$f" in
    *.tar.gz) mkdir -p "$dl/$name.x" && tar -xzf "$f" -C "$dl/$name.x" && install -m 0755 "$(find "$dl/$name.x" -type f -name "$name" | head -1)" "$CI_BIN/$name" ;;
    *.py)     install -m 0644 "$f" "$CI_BIN/$name.py" ;;
    *.yaml)   install -m 0644 "$f" "$CI_BIN/$name.yaml" ;;
    *)        echo "no install rule for $name" >&2; exit 1 ;;
  esac
  echo "ok $name ${sha:0:12}"
done < ci/tools.lock
python3 -m venv "$CI_BIN/venv"
"$CI_BIN/venv/bin/pip" install --quiet --disable-pip-version-check \
  --require-hashes --only-binary :all: -r requirements-dev.lock
echo "ok python dependencies (requirements-dev.lock, every hash checked)"
