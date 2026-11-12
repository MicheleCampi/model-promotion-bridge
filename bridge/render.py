"""Turn a promoted model into the manifest a cluster applies.

Emits YAML rather than calling the Kubernetes API, and that is the whole
design. A manifest is reviewable before it lands, diffable against what is
running, and replayable from git months later; an API call is none of
those. It also means the bridge needs no cluster credentials, which is
one fewer thing to hold.
"""
from __future__ import annotations

from typing import Any, Optional

import yaml

from .resolve import PromotedModel

# The operator has no first-class field for the model revision: VllmServiceSpec
# carries `model: String` and nothing beside it. It does have extraArgs, which
# passes arguments straight to `vllm serve`, and vLLM accepts --revision
# (vllm/engine/arg_utils.py:950 at vLLM 84bcbc6). The tokenizer inherits that
# revision unless --tokenizer-revision is given (vllm/config/model.py:590-591).
# So the pin travels as an extra arg, and the annotations below record where the
# value came from — because an extra arg alone tells a reader what was served
# but not who decided it.
PROVENANCE_PREFIX = "model.michelecampi.dev"

# Flags that would compete with the pin: a second --revision makes the
# effective pin depend on argument order, and a --tokenizer-revision serves the
# pinned weights with a tokenizer from elsewhere. Matched as a separate token
# and in the one-token `--flag=value` form.
_PIN_FLAGS = ("--revision", "--tokenizer-revision")

# The two kinds an AppProject of a team admits (gke argocd/apps/team-*.yaml).
KINDS = ("VllmService", "FleetService")


def build_manifest(
    promoted: PromotedModel,
    service_name: str,
    namespace: str = "default",
    replicas: int = 1,
    gpu: int = 1,
    image: Optional[str] = None,
    cpu: Optional[str] = None,
    memory: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    kind: str = "VllmService",
    node_selector: Optional[dict[str, str]] = None,
    ttft_seconds: Optional[float] = None,
    percentile: Optional[float] = None,
    budget_eur: Optional[float] = None,
    provision_on_empty: bool = False,
) -> dict[str, Any]:
    """Build a VllmService that serves exactly the promoted revision.

    The revision goes into extraArgs where vLLM will read it, and into
    annotations where a human or an audit will. Both, not either: the arg
    is what takes effect and the annotation is what survives a reader who
    does not know vLLM's flags.
    """
    args = list(extra_args or [])
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in _PIN_FLAGS:
            raise ValueError(
                f"extra_args already carries {flag}; the bridge owns the "
                f"revision pin, and a competing flag would either make which "
                f"pin takes effect depend on argument order or split the "
                f"tokenizer from the weights"
            )
    args += ["--revision", promoted.revision]

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if node_selector and kind != "FleetService":
        raise ValueError("node_selector is read only by FleetService (spec.nodePool.selector)")
    if provision_on_empty and kind != "FleetService":
        raise ValueError("provision_on_empty is read only by FleetService (spec.nodePool.provisionOnEmpty)")
    if percentile is not None and ttft_seconds is None:
        raise ValueError("percentile without ttft_seconds is not an objective")
    pod: dict[str, Any] = {"gpu": gpu, "extraArgs": args}
    if image:
        pod["image"] = image
    # Without these the pod takes the namespace LimitRange defaults; in a
    # tenant namespace of the gke repo that is a 512Mi memory limit, which a
    # vLLM process exceeds while loading. Emitted only when given, so a
    # promotion rendered without them is an unchanged manifest.
    if cpu:
        pod["cpu"] = cpu
    if memory:
        pod["memory"] = memory
    # FleetServiceSpec (operator v0.9.0) requires model, replicas and
    # template; what each pod runs lives in template. VllmServiceSpec keeps
    # the same fields at the top level.
    spec: dict[str, Any] = {"model": promoted.hf_repo, "replicas": replicas}
    if kind == "FleetService":
        spec["template"] = pod
        pool: dict[str, Any] = {}
        if node_selector:
            pool["selector"] = dict(node_selector)
        # operator v0.10.0: with no candidate node, a placeholder Pod asks the
        # cluster autoscaler for one (D-0145).
        if provision_on_empty:
            pool["provisionOnEmpty"] = True
        if pool:
            spec["nodePool"] = pool
    else:
        spec.update(pod)
    # The team's declaration (ADR-0013, ADR-0025). Both kinds carry it from
    # operator v0.10.0; a fleet copies it to every replica. Emitted only when
    # given; percentile defaults to the platform's 95.
    if ttft_seconds is not None:
        spec["objective"] = {
            "ttftSeconds": ttft_seconds,
            "percentile": 95.0 if percentile is None else percentile,
        }
    if budget_eur is not None:
        spec["budget"] = {"eurPerMonth": budget_eur}

    return {
        "apiVersion": "inference.michelecampi.dev/v1alpha1",
        "kind": kind,
        "metadata": {
            "name": service_name,
            "namespace": namespace,
            "annotations": {
                f"{PROVENANCE_PREFIX}/registry-name": promoted.name,
                f"{PROVENANCE_PREFIX}/registry-version": promoted.version,
                f"{PROVENANCE_PREFIX}/registry-alias": promoted.alias,
                f"{PROVENANCE_PREFIX}/source": promoted.source,
                f"{PROVENANCE_PREFIX}/revision": promoted.revision,
            },
        },
        "spec": spec,
    }


def render(manifest: dict[str, Any], header: Optional[str] = None) -> str:
    """Serialise deterministically, so an unchanged promotion is an empty diff.

    sort_keys=True is not cosmetic here: this output is committed to git and
    reconciled by ArgoCD, and a manifest whose key order drifts between runs
    produces spurious diffs that train a reviewer to ignore them.
    """
    body = yaml.safe_dump(manifest, sort_keys=True, default_flow_style=False)
    if header:
        prefix = "\n".join(f"# {line}" for line in header.splitlines())
        return f"{prefix}\n{body}"
    return body
