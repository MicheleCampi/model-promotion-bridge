"""What the bridge must refuse, and what it must reproduce.

The refusals carry more weight than the happy path here. A bridge that
emits a manifest when the registry cannot say what to serve produces a
deployment that looks reviewed and is not, and that failure is silent all
the way to production.
"""
import socket
import sys
from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge import cli
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


def _closed_port():
    """A local port with nothing listening on it: bind, read it, release it."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _no_retries(monkeypatch):
    """Fail fast on a refused connection instead of retrying with backoff."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")


def test_refuses_the_reserved_latest_alias(registry):
    """MLflow resolves the alias "latest", in any case, to the newest version.

    The store's own behaviour is asserted first, so the test shows where the
    hazard comes from (mlflow 3.15.1, store/model_registry/sqlalchemy_store.py:1571).
    Followed blindly, it would serve the newest registration instead of the
    promoted one.
    """
    uri, c = registry
    v1 = _version(c, {"hf_revision": REV})
    v2 = _version(c, {"hf_revision": "b" * 40})
    c.set_registered_model_alias("qwen2.5-7b-instruct", "production", v1.version)
    assert c.get_model_version_by_alias("qwen2.5-7b-instruct", "latest").version == v2.version
    for alias in ("latest", "LATEST"):
        with pytest.raises(PromotionError, match="reserved"):
            resolve("qwen2.5-7b-instruct", alias, uri)


def test_an_unreachable_registry_is_not_reported_as_nothing_promoted(monkeypatch):
    """An outage must not look like "nothing is promoted".

    MLflow's real failure path, not a mock: the REST client turns a refused
    connection into an MlflowException with no error code (mlflow 3.15.1,
    utils/rest_utils.py:317), which defaults to INTERNAL_ERROR
    (exceptions.py:79). A pipeline that stops quietly on "nothing promoted"
    must not also stop quietly on an outage.
    """
    _no_retries(monkeypatch)
    with pytest.raises(Exception) as caught:
        resolve("qwen2.5-7b-instruct", "production", f"http://127.0.0.1:{_closed_port()}")
    assert not isinstance(caught.value, PromotionError), "outage reported as nothing promoted"
    from bridge.resolve import RegistryUnavailable
    assert isinstance(caught.value, RegistryUnavailable)


def test_cli_exit_codes_tell_the_outcomes_apart(registry, monkeypatch, capsys):
    """0 manifest written, 3 nothing promoted, 4 registry unreachable.

    1 and 2 are left to Python (an uncaught exception) and argparse (a usage
    error), so no outcome shares an exit code with a crash.
    """
    uri, c = registry
    base = ["--model", "qwen2.5-7b-instruct", "--service-name", "svc"]
    assert cli.main(["--tracking-uri", uri, *base]) == 3
    v = _version(c, {"hf_revision": REV})
    c.set_registered_model_alias("qwen2.5-7b-instruct", "production", v.version)
    capsys.readouterr()
    assert cli.main(["--tracking-uri", uri, *base]) == 0
    assert f"- {REV}" in capsys.readouterr().out
    _no_retries(monkeypatch)
    assert cli.main(["--tracking-uri", f"http://127.0.0.1:{_closed_port()}", *base]) == 4


def test_refuses_a_second_revision_flag_in_equals_form():
    """`--revision=X` is the same flag written as one token."""
    p = PromotedModel("m", "1", "hf://X/Y", REV, "production")
    with pytest.raises(ValueError, match="already carries --revision"):
        build_manifest(p, "svc", extra_args=["--revision=deadbeef"])


def test_refuses_a_tokenizer_revision_that_would_split_the_pin():
    """The tokenizer inherits --revision unless told otherwise.

    vLLM sets tokenizer_revision from revision when it is not given
    (vllm/config/model.py:590-591 at vLLM 84bcbc6). A --tokenizer-revision in
    extra_args would serve the pinned weights with a tokenizer from elsewhere.
    """
    p = PromotedModel("m", "1", "hf://X/Y", REV, "production")
    with pytest.raises(ValueError, match="already carries --tokenizer-revision"):
        build_manifest(p, "svc", extra_args=["--tokenizer-revision", "main"])
