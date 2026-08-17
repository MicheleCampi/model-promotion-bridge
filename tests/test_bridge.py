"""What the bridge must refuse, and what it must reproduce.

The refusals carry more weight than the happy path here. A bridge that
emits a manifest when the registry cannot say what to serve produces a
deployment that looks reviewed and is not, and that failure is silent all
the way to production.
"""
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.render import build_manifest, render
from bridge.resolve import PromotedModel, PromotionError, resolve

REV = "a09a35458c702b33eeacc393d103063234e8bc28"


@pytest.fixture()
def registry(tmp_path):
    """A real MLflow registry on sqlite, not a mock.

    Mocking the client would test my idea of the API rather than the API:
    MLflow 3.15 put the file store into maintenance mode and raises on it,
    which a mock would have hidden until the first real run.
    """
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    c = MlflowClient()
    c.create_registered_model("qwen2.5-7b-instruct")
    return uri, c


def _version(client, tags):
    return client.create_model_version(
        name="qwen2.5-7b-instruct",
        source="hf://Qwen/Qwen2.5-7B-Instruct",
        tags=tags,
    )


def test_resolves_the_alias_not_the_latest_version(registry):
    """Promotion is what the alias says, not what was registered last."""
    uri, c = registry
    v1 = _version(c, {"hf_revision": REV})
    v2 = _version(c, {"hf_revision": "b" * 40})
    c.set_registered_model_alias("qwen2.5-7b-instruct", "production", v1.version)

    got = resolve("qwen2.5-7b-instruct", "production", uri)
    assert got.version == str(v1.version)
    assert got.revision == REV
    assert v2.version != v1.version  # a newer version exists and is not served


def test_refuses_when_the_alias_does_not_exist(registry):
    uri, _ = registry
    with pytest.raises(PromotionError, match="no alias"):
        resolve("qwen2.5-7b-instruct", "production", uri)


def test_refuses_a_promoted_version_with_no_revision(registry):
    """The failure this whole thing exists to prevent.

    A version promoted without a pin must stop the pipeline. Emitting a
    manifest that omits --revision would deploy whatever HEAD of the model
    repo happens to be, which is precisely the unreproducible state the
    bridge is meant to close.
    """
    uri, c = registry
    v = _version(c, {"engine": "vllm"})
    c.set_registered_model_alias("qwen2.5-7b-instruct", "production", v.version)
    with pytest.raises(PromotionError, match="no 'hf_revision' tag"):
        resolve("qwen2.5-7b-instruct", "production", uri)


def test_revision_reaches_both_the_flag_and_the_annotation():
    p = PromotedModel("m", "3", "hf://Qwen/Qwen2.5-7B-Instruct", REV, "production")
    m = build_manifest(p, "svc")
    args = m["spec"]["extraArgs"]
    assert args[args.index("--revision") + 1] == REV
    assert m["metadata"]["annotations"]["model.michelecampi.dev/revision"] == REV


def test_refuses_a_second_revision_flag():
    """Two --revision flags make the effective pin depend on argument order."""
    p = PromotedModel("m", "1", "hf://X/Y", REV, "production")
    with pytest.raises(ValueError, match="already carries --revision"):
        build_manifest(p, "svc", extra_args=["--revision", "deadbeef"])


def test_render_is_stable_across_runs():
    """An unchanged promotion must produce a byte-identical manifest.

    This output is committed and reconciled; a rendering that reorders keys
    between runs produces diffs with no change behind them, and a reviewer
    who sees those learns to skim.
    """
    p = PromotedModel("m", "1", "hf://X/Y", REV, "production")
    a = render(build_manifest(p, "svc"))
    b = render(build_manifest(p, "svc"))
    assert a == b
    assert yaml.safe_load(a)["spec"]["model"] == "X/Y"
