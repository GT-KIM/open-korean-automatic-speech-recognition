"""Optional immutable Hub revisions, shared by model and processor loaders."""
import re


def pinned_revisions(config):
    model = getattr(config, "model_revision", None)
    processor = getattr(config, "processor_revision", None) or model
    if processor and not model:
        raise ValueError("processor_revision requires model_revision")
    for name, value in (("model_revision", model), ("processor_revision", processor)):
        if value is not None and not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"{name} must be a full lowercase 40-character commit hash")
    return model, processor


def verify_loaded_revision(model, expected):
    actual = getattr(getattr(model, "config", None), "_commit_hash", None)
    if expected is not None and actual != expected:
        raise ValueError("Loaded model revision does not match the pinned commit")
    return actual


def verify_processor_revision(processor, expected):
    actual = getattr(processor, "_commit_hash", None)
    if not actual:
        tokenizer = getattr(processor, "tokenizer", None)
        actual = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if expected and actual and actual != expected:
        raise ValueError("Loaded processor revision does not match the pinned commit")
    # Many processors omit their resolved hash. In that case the immutable loader
    # argument is the evidence, not a claimed observation of loaded metadata.
    return "loaded_metadata" if actual else ("pinned_loader_argument" if expected else None)
