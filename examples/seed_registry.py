#!/usr/bin/env python3
"""Populate a registry the way a platform that serves third-party models would.

Two versions of the same model, two different HuggingFace revisions, and
an alias pointing at one of them. That is the whole state a promotion
decision consists of, and it is enough to demonstrate the property that
matters: the manifest follows the alias, not the newest version.

The revisions are real commit SHAs from the model repo. The first is the
one a GPU session on 2026-07-04 actually served — recorded by hand in that
session's provenance file, because at the time there was nowhere else to
put it. The second is an earlier commit of the same repo, registered after
the first and left unpromoted: the order of registration does not decide
what is served, the alias does.
"""
import argparse

import mlflow
from mlflow import MlflowClient

MODEL = "qwen2.5-7b-instruct"
SOURCE = "hf://Qwen/Qwen2.5-7B-Instruct"
SERVED_2026_07_04 = "a09a35458c702b33eeacc393d103063234e8bc28"
EARLIER = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", required=True)
    a = p.parse_args()

    mlflow.set_tracking_uri(a.tracking_uri)
    c = MlflowClient()
    try:
        c.create_registered_model(
            MODEL, description="Third-party weights served by vllm-coldstart-operator")
    except Exception:
        pass  # already there: re-running must not be destructive

    v1 = c.create_model_version(MODEL, SOURCE, tags={
        "hf_revision": SERVED_2026_07_04,
        "engine": "vllm",
        "note": "revision served in the 3xA10 fleet session of 2026-07-04",
    })
    v2 = c.create_model_version(MODEL, SOURCE, tags={
        "hf_revision": EARLIER,
        "engine": "vllm",
        "note": "earlier upstream revision, registered after v1, not promoted",
    })
    c.set_registered_model_alias(MODEL, "production", v1.version)

    print(f"registered {MODEL}: v{v1.version} (promoted) and v{v2.version}")
    print(f"  production -> v{v1.version}, revision {SERVED_2026_07_04[:12]}")
    print(f"  newest     -> v{v2.version}, revision {EARLIER[:12]} (registered last, not served)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
