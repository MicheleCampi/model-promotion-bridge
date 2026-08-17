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
# passes arguments straight to `vllm serve`, and vLLM accepts --revision. So the
# pin travels as an extra arg, and the annotations below record where the value
# came from — because an extra arg alone tells a reader what was served but not
# who decided it.
PROVENANCE_PREFIX = "model.michelecampi.dev"


def build_manifest(
    promoted: PromotedModel,
    service_name: str,
    namespace: str = "default",
    replicas: int = 1,
    gpu: int = 1,
    image: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a VllmService that serves exactly the promoted revision.

    The revision goes into extraArgs where vLLM will read it, and into
    annotations where a human or an audit will. Both, not either: the arg
    is what takes effect and the annotation is what survives a reader who
    does not know vLLM's flags.
    """
    args = list(extra_args or [])
    if "--revision" in args:
        raise ValueError(
            "extra_args already carries --revision; the bridge owns that flag "
            "and a second one would make which pin takes effect depend on "
            "argument order"
        )
    args += ["--revision", promoted.revision]

    spec: dict[str, Any] = {
        "model": promoted.hf_repo,
        "replicas": replicas,
        "gpu": gpu,
        "extraArgs": args,
    }
    if image:
        spec["image"] = image

    return {
        "apiVersion": "inference.michelecampi.dev/v1alpha1",
        "kind": "VllmService",
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
