"""Resolve what a model registry has promoted into what a cluster should serve.

The gap this closes is one my own evidence recorded before I had a way to
fill it. A GPU session from 2026-07-04 left a provenance file that
recorded, next to the driver and cluster versions, the container image
pinned by digest and the model snapshot as a bare commit SHA, added by
hand: the session checklist asked for the digest, not for the model. The
FleetService that served it declared only `Qwen/Qwen2.5-7B-Instruct` — a
name, not a revision. One artefact in that pod was reproducible and the other was
not, and the difference was invisible from the manifest.

So: the registry owns which revision is current, GitOps applies the
manifest, and the operator serves it. Three concerns, kept apart on
purpose. The bridge does not talk to the cluster and the operator does
not talk to the registry — an operator that reached into MLflow would be
a platform component coupled to one vendor's tool, and the manifest is a
cheaper contract than an integration.

What this deliberately is NOT: a training pipeline. Nothing here trains,
fine-tunes or evaluates. The models are third-party weights served as
they are, and the registry describes their provenance rather than
claiming authorship. That is the honest shape for a platform that serves
models it did not produce, and pretending otherwise would be the kind of
artefact that exists to fill a checkbox.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

# The error codes with which the registry answers "there is nothing here":
# no such alias and no such model (mlflow 3.15.1,
# store/model_registry/sqlalchemy_store.py:1591-1593 and :439-442). A REST
# client receives the same code from the server's JSON (exceptions.py:161).
# No HTTP status maps to either (exceptions.py:54-57), so a proxy's 404 or
# 400 is not mistaken for an empty registry. Every other code is an outage.
_NOTHING_TO_SERVE = frozenset(
    {ErrorCode.Name(INVALID_PARAMETER_VALUE), ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)}
)

# MLflow resolves this alias, in any letter case, to the newest registered
# version rather than to a promotion (mlflow 3.15.1, utils/validation.py:57,
# store/model_registry/sqlalchemy_store.py:1571).
_RESERVED_ALIAS = "latest"


class PromotionError(RuntimeError):
    """Raised when the registry answers that there is nothing to serve.

    No such alias or model, the reserved alias "latest", or a promoted
    version without a revision tag. The registry did answer, so this is a
    state a pipeline can stop on quietly. An outage raises
    RegistryUnavailable instead.
    """


class RegistryUnavailable(RuntimeError):
    """Raised when the registry could not answer at all.

    Deliberately not a subclass of PromotionError: code written to stop
    quietly on "nothing promoted" must not swallow an outage it never
    anticipated. Any error code outside _NOTHING_TO_SERVE lands here, so a
    code this module has never seen is treated as an outage, not as an
    empty registry.
    """


@dataclasses.dataclass(frozen=True)
class PromotedModel:
    """What the registry says is current, with everything needed to serve it.

    Frozen because this crosses the boundary between the registry and the
    manifest: a value that could be edited in flight would let the
    generated YAML drift from what the registry actually holds, which is
    the one property this whole thing exists to guarantee.
    """

    name: str
    version: str
    source: str
    revision: Optional[str]
    alias: str

    @property
    def hf_repo(self) -> str:
        """The HuggingFace repo id, stripped of the hf:// scheme.

        The registry stores a URI because a source may not be HuggingFace;
        vLLM wants a bare repo id. Converting here rather than at the call
        site keeps the assumption in one place.
        """
        if self.source.startswith("hf://"):
            return self.source[len("hf://"):]
        return self.source


def resolve(model_name: str, alias: str, tracking_uri: str,
            revision_tag: str = "hf_revision") -> PromotedModel:
    """Read the registry and return what the alias currently points at.

    Absence is an error, not a default. If the alias does not exist, or
    the promoted version carries no revision tag, this raises rather than
    falling back to "latest" — a pipeline that silently deploys an
    unpinned model when the pin is missing defeats the point of pinning.

    Raises PromotionError when the registry answers that there is nothing
    to serve, and RegistryUnavailable when it cannot answer.
    """
    if alias.lower() == _RESERVED_ALIAS:
        raise PromotionError(
            f"alias {alias!r} is reserved: MLflow resolves it to the newest "
            f"registered version, not to a promotion. Use the alias a "
            f"promotion was recorded under."
        )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        if exc.error_code in _NOTHING_TO_SERVE:
            raise PromotionError(
                f"registry has no alias {alias!r} on model {model_name!r} "
                f"({tracking_uri}): {exc}"
            ) from exc
        raise RegistryUnavailable(
            f"registry at {tracking_uri} could not answer for "
            f"{model_name}@{alias} ({exc.error_code}): {exc}"
        ) from exc

    revision = (mv.tags or {}).get(revision_tag)
    if not revision:
        raise PromotionError(
            f"{model_name}@{alias} is version {mv.version} but carries no "
            f"{revision_tag!r} tag. Refusing to emit a manifest without a "
            f"pin: an unpinned deployment is what this bridge exists to "
            f"prevent."
        )
    return PromotedModel(
        name=model_name,
        version=str(mv.version),
        source=mv.source,
        revision=revision,
        alias=alias,
    )
