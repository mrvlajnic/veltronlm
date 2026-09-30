"""VeltronLM: a real open-weight ~4B-parameter language model and customer-support system.

This package contains a complete, self-contained implementation:

* :mod:`veltron.model`      -- decoder-only Transformer (RoPE, GQA, RMSNorm, SwiGLU, KV cache)
* :mod:`veltron.tokenizer`  -- byte-level BPE tokenizer trained from scratch
* :mod:`veltron.data`       -- dataset acquisition, quality pipeline, sharding
* :mod:`veltron.training`   -- pretraining loop with checkpoint/resume
* :mod:`veltron.finetuning` -- supervised instruction tuning
* :mod:`veltron.alignment`  -- direct preference optimisation
* :mod:`veltron.rag`        -- retrieval augmented generation
* :mod:`veltron.support`    -- customer-support specialization
* :mod:`veltron.inference`  -- generation engine, KV cache, sampling, quantization
* :mod:`veltron.evaluation` -- benchmark and metric framework
* :mod:`veltron.serving`    -- HTTP API
* :mod:`veltron.chatbot`    -- web chat UI

Nothing in this package calls an external model API. The tokenizer, the weights and
the logits are all produced locally.
"""

__version__ = "0.1.0-alpha"

MODEL_FAMILY = "VeltronLM"

__all__ = ["__version__", "MODEL_FAMILY"]
