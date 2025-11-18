# src/utils/wandb_utils.py

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

wandb = _wandb  # re-export for convenience

# repo root (two levels up from this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _read_api_key(secrets_path: str):
    """Read the W&B API key from a text file containing only the key."""
    try:
        with open(secrets_path, "r") as f:
            key = f.read().strip()
        if not key:
            logger.warning("W&B: .secrets file is empty; disabling W&B logging.")
            return None
        return key
    except FileNotFoundError:
        logger.warning("W&B: secrets file '%s' not found; disabling W&B logging.", secrets_path)
        return None
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("W&B: error reading secrets file '%s': %s; disabling W&B logging.", secrets_path, e)
        return None


def _auto_naming_and_tags(config: dict):
    """
    Build a base slug and a set of default tags from the training config.

    slug (no timestamp) is used as:
      - base for run_name (we append a timestamp)

    Returns:
      slug: str
      tags: List[str]
    """
    logging_cfg = config.get("logging", {})
    meta_cfg = config.get("meta", {})
    data_cfg = config.get("data", {})
    mask_cfg = config.get("mask", {})
    opt_cfg = config.get("optimization", {})

    write_tag = logging_cfg.get("write_tag", "ijepa")
    model_name = meta_cfg.get("model_name", "model")
    crop_size = data_cfg.get("crop_size", None)
    patch_size = mask_cfg.get("patch_size", None)
    epochs = opt_cfg.get("epochs", None)
    image_folder = data_cfg.get("image_folder", "")

    dataset_slug = None
    if image_folder:
        dataset_slug = os.path.basename(str(image_folder).rstrip("/")) or None

    parts = [write_tag, model_name]
    if dataset_slug:
        parts.append(dataset_slug)
    if crop_size is not None:
        parts.append(f"cs{crop_size}")
    if patch_size is not None:
        parts.append(f"ps{patch_size}")
    if epochs is not None:
        parts.append(f"ep{epochs}")

    slug = "-".join(str(p) for p in parts if p)

    # Default tags (you can still add your own from config)
    tags = ["ijepa"]
    if write_tag:
        tags.append(write_tag)
    if model_name:
        tags.append(model_name)
    if dataset_slug:
        tags.append(dataset_slug)
    if crop_size is not None:
        tags.append(f"{crop_size}px")
    if patch_size is not None:
        tags.append(f"patch{patch_size}")
    if epochs is not None:
        tags.append(f"{epochs}ep")

    # Deduplicate preserving order
    seen = set()
    uniq_tags = []
    for t in tags:
        if t not in seen:
            uniq_tags.append(t)
            seen.add(t)

    return slug, uniq_tags


def init_wandb(config: dict, rank: int):
    """
    Initialize a Weights & Biases run from the provided config dict.

    - Reads API key from a .secrets file (path configurable in config).
    - Only rank 0 creates a W&B run; other ranks return None.
    - project/entity/mode come from config['logging']['wandb'].
    - run_name and tags are auto-generated if not specified.
    """
    if wandb is None:
        logger.info("W&B not installed; skipping W&B logging.")
        return None

    logging_cfg = config.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})

    if not wandb_cfg.get("enable", False):
        return None

    # Only log from rank 0 to avoid duplicated runs
    if rank != 0:
        # Make sure other ranks don't try to talk to W&B
        os.environ.setdefault("WANDB_MODE", "disabled")
        return None

    # Resolve secrets file path
    secrets_rel = wandb_cfg.get("secrets_file", ".secrets")
    if os.path.isabs(secrets_rel):
        secrets_path = secrets_rel
    else:
        secrets_path = os.path.join(ROOT_DIR, secrets_rel)

    api_key = _read_api_key(secrets_path)
    if api_key is None:
        return None

    try:
        wandb.login(key=api_key)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("W&B: failed to login: %s; disabling W&B logging.", e)
        return None

    # Required / semi-required fields
    project = wandb_cfg.get("project", "ijepa")
    entity = wandb_cfg.get("entity", None)

    # Auto naming + tags
    base_slug, auto_tags = _auto_naming_and_tags(config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Allow manual override from config, otherwise fall back to auto
    name = wandb_cfg.get("run_name") or f"{base_slug}-{timestamp}"

    user_tags = wandb_cfg.get("tags") or []
    if isinstance(user_tags, str):
        user_tags = [user_tags]

    # Combine user tags + auto tags (dedup, preserve order)
    all_tags = []
    seen = set()
    for t in list(user_tags) + list(auto_tags):
        if t not in seen:
            all_tags.append(t)
            seen.add(t)

    mode = wandb_cfg.get("mode", None)  # e.g. 'offline'
    if mode is not None:
        os.environ["WANDB_MODE"] = str(mode)

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=all_tags,
        config=config,
    )

    # Define a global step metric and tie all train/* metrics to it
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")

    logger.info("Initialized W&B run: project=%s, entity=%s, name=%s",
                project, entity, name)
    logger.info("W&B tags: %s", all_tags)

    return run
