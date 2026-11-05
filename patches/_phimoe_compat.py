# -*- coding: utf-8 -*-
"""Compatibility layer for DAOP's PhiMoE import.

DAOP imports PhiMoE symbols from `transformers.models.phimoe`, which does not
exist in transformers 4.44, the version DAOP runs on. The Mixtral classes live
in the same files, so the import fails before any Mixtral work can start.

This module supplies the names DAOP needs. It re-exports the transformers
symbols directly, and for the PhiMoE classes it either loads Microsoft's
published remote code (set DAOP_PHIMOE_DIR to the model directory) or falls
back to a stub. The stub is enough for the Mixtral path, which is what the
paper evaluates.
"""
import importlib as _il
import os as _os
import sys as _sys
import types as _types

from transformers.modeling_outputs import MoeModelOutputWithPast  # noqa: F401
from transformers.modeling_attn_mask_utils import (  # noqa: F401
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

_PHI_DIR = _os.environ.get("DAOP_PHIMOE_DIR", "")
_PKG = "_phimoe_remote"


def _load_real():
    pkg = _types.ModuleType(_PKG)
    pkg.__path__ = [_PHI_DIR]          # 상대 import 가 이 디렉토리 안에서 해결되게 한다
    _sys.modules[_PKG] = pkg
    cfg = _il.import_module(_PKG + ".configuration_phimoe")
    mdl = _il.import_module(_PKG + ".modeling_phimoe")
    return cfg.PhiMoEConfig, mdl.PhiMoEPreTrainedModel, mdl.sparsemixer


if _os.path.exists(_os.path.join(_PHI_DIR, "modeling_phimoe.py")):
    PhiMoEConfig, PhiMoEPreTrainedModel, sparsemixer = _load_real()
else:
    from transformers.configuration_utils import PretrainedConfig

    class PhiMoEConfig(PretrainedConfig):
        model_type = "phimoe"

    class PhiMoEPreTrainedModel:
        def __init__(self, *a, **k):
            raise NotImplementedError(
                "PhiMoE 원본 remote code 를 찾지 못했다: " + _PHI_DIR)

    def sparsemixer(*a, **k):
        raise NotImplementedError("sparsemixer 는 PhiMoE 전용이다.")
