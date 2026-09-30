"""Autoregressive generation.

Implements greedy, temperature, top-k, top-p (nucleus), min-p and repetition-penalty
sampling over a KV cache, with streaming token output.

Logits for decode steps are computed in fp32 even when the model runs in fp16. A 65536-way
logit vector in fp16 saturates at 65504, so a confident prediction overflows to ``inf``
and then to ``nan`` in the softmax -- a failure that only appears at exactly 4B scale and
only for the highest-probability tokens, which is where it matters most.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..model.layers import KVCache
from ..model.transformer import VeltronLM
from ..utils.device import get_device, probe, synchronize
from ..utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class GenerationConfig:
    """Sampling parameters."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0                # 0 disables
    top_p: float = 1.0            # 1.0 disables
    min_p: float = 0.0            # 0 disables
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    stop_strings: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    greedy: bool = False
    num_return_sequences: int = 1
    max_batch_size: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationConfig:
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in d.items() if k in known}
        for key in ("stop_strings", "stop_token_ids"):
            if key in clean and clean[key] is not None and not isinstance(clean[key], tuple):
                clean[key] = tuple(clean[key])
        return cls(**clean)


@dataclass
class GenerationResult:
    """One completed generation plus the statistics needed to report its cost."""

    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    seconds: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0
    peak_memory_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sampler:
    """Token selection strategies, each a separate, individually testable step."""

    def __init__(self, cfg: GenerationConfig, vocab_size: int, device: torch.device) -> None:
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.device = device
        self.generator = None
        if cfg.seed is not None:
            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(cfg.seed)

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self.generator is not None:
            return torch.rand(shape, generator=self.generator, dtype=torch.float32).to(self.device)
        return torch.rand(shape, dtype=torch.float32, device=self.device)

    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor,
        generated: Sequence[int],
        penalty: float,
        presence: float = 0.0,
        frequency: float = 0.0,
    ) -> torch.Tensor:
        """Standard CTRL-style repetition penalty plus presence/frequency terms.

        Positive logits are divided and negative logits multiplied, so the penalty pushes
        already-emitted tokens away in both directions rather than clamping.
        """
        if penalty == 1.0 and presence == 0.0 and frequency == 0.0:
            return logits
        if not generated:
            return logits
        out = logits.clone()
        idx = torch.tensor(sorted(set(generated)), device=logits.device, dtype=torch.long)
        selected = out.index_select(-1, idx)
        if penalty != 1.0:
            selected = torch.where(selected > 0, selected / penalty, selected * penalty)
        if presence:
            selected = selected - presence
        if frequency:
            counts = torch.zeros_like(selected)
            for t in set(generated):
                counts[:, idx.tolist().index(t)] += 1
            selected = selected - frequency * counts
        out.index_copy_(-1, idx, selected)
        return out

    @staticmethod
    def no_repeat_ngram(logits: torch.Tensor, generated: Sequence[int], n: int) -> torch.Tensor:
        """Mask tokens that would complete an n-gram already present in the output."""
        if n <= 0 or len(generated) < n:
            return logits
        prefix = tuple(generated[-(n - 1):]) if n > 1 else ()
        banned: set[int] = set()
        for i in range(len(generated) - n + 1):
            if tuple(generated[i : i + n - 1]) == prefix:
                banned.add(generated[i + n - 1])
        if not banned:
            return logits
        out = logits.clone()
        out[:, torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)] = -math.inf
        return out

    def select(self, logits: torch.Tensor, generated: Sequence[Sequence[int]]) -> torch.Tensor:
        """Return selected token ids of shape ``(B,)`` from ``(B, V)`` logits."""
        cfg = self.cfg
        B, V = logits.shape

        if cfg.repetition_penalty != 1.0 or cfg.presence_penalty or cfg.frequency_penalty:
            for b in range(B):
                if generated[b]:
                    logits[b] = self.apply_repetition_penalty(
                        logits[b], generated[b], cfg.repetition_penalty,
                        cfg.presence_penalty, cfg.frequency_penalty,
                    )
        if cfg.no_repeat_ngram_size > 0:
            for b in range(B):
                if generated[b]:
                    logits[b] = self.no_repeat_ngram(logits[b], generated[b], cfg.no_repeat_ngram_size)

        if cfg.greedy:
            return logits.argmax(dim=-1)

        # Temperature 0 is equivalent to greedy; anything <=0 is a config error we treat
        # as greedy rather than dividing by zero.
        temp = max(cfg.temperature, 1e-6)
        probs = torch.softmax(logits / temp, dim=-1)

        if cfg.min_p > 0.0:
            # Keep tokens with prob >= min_p * prob(max). Unlike top_p this is scale-free
            # and works better when the distribution is flat.
            top = probs.max(dim=-1, keepdim=True).values
            probs = torch.where(probs < (cfg.min_p * top), torch.zeros_like(probs), probs)

        if cfg.top_k and cfg.top_k > 0:
            k = min(cfg.top_k, V)
            kth = probs.topk(k, dim=-1).values[:, -1:]
            probs = torch.where(probs < kth, torch.zeros_like(probs), probs)

        if cfg.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Keep the first token whose cumulative mass exceeds top_p, always keeping at
            # least one token so the distribution never goes empty.
            remove = cumulative - sorted_probs > cfg.top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

        total = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(total > 0, probs / total.clamp_min(1e-20), torch.zeros_like(probs))

        u = self._rand((B, V))
        return torch.multinomial(probs.cpu(), num_samples=1, generator=self.generator).squeeze(-1).to(self.device)


class Generator:
    """High-level generation with KV-cache reuse across calls."""

    def __init__(
        self,
        model: VeltronLM,
        tokenizer: Any,
        device: str | torch.device = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device_info = probe(str(device)) if isinstance(device, str) else None
        self.device = get_device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        if dtype is not None:
            self.model.to(dtype)
        self.model.eval()
        self.dtype = dtype
        self._cache: KVCache | None = None
        self._cache_len = 0

    # ------------------------------------------------------------------ cache
    def _ensure_cache(self, batch: int, needed: int, dtype: torch.dtype) -> KVCache:
        cache_dtype = dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        if self._cache is None or self._cache.batch != batch or self._cache.dtype != cache_dtype:
            self._cache = self.model.init_kv_cache(
                batch=batch, max_seq_len=max(needed, 512),
                dtype=cache_dtype, device=self.device,
            )
            self._cache_len = 0
        elif self._cache.max_seq_len < needed * 2:
            self._cache.grow_to(max(needed * 2, self._cache.max_seq_len))
        return self._cache

    def reset_cache(self) -> None:
        self._cache = None
        self._cache_len = 0

    # ------------------------------------------------------------- generation
    def _prepare_batch(
        self,
        prompt_ids: Sequence[int] | Sequence[Sequence[int]],
        cfg: GenerationConfig,
        cache: KVCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Normalise prompts into left-padded ids, RoPE positions and per-row pad counts."""
        if prompt_ids and isinstance(prompt_ids[0], int):
            prompts = [[int(t) for t in prompt_ids]]  # type: ignore[arg-type]
        else:
            prompts = [list(map(int, p)) for p in prompt_ids]  # type: ignore[union-attr]

        B = len(prompts)
        if B > cfg.max_batch_size:
            raise ValueError(f"batch size {B} exceeds max_batch_size={cfg.max_batch_size}")

        # Left-pad so every sequence's final real token sits at the same index; otherwise
        # a batched generation would continue from pad tokens for the shorter prompts.
        maxlen = max(len(p) for p in prompts)
        pad_id = getattr(self.tokenizer, "pad_id", 0) or 0
        padded = [[pad_id] * (maxlen - len(p)) + p for p in prompts]
        lengths = [len(p) for p in prompts]

        ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        # RoPE positions must start at each sequence's true token index, not at 0, or
        # short prompts get the wrong rotary frequencies.
        pos = torch.arange(maxlen, device=self.device).unsqueeze(0).expand(B, maxlen)
        pos = pos - (maxlen - torch.tensor(lengths, device=self.device)).unsqueeze(1)
        pos = pos.clamp_min(0)
        mask = torch.zeros(B, maxlen, dtype=torch.bool, device=self.device)
        for b in range(B):
            mask[b, maxlen - lengths[b]:] = True

        pad_counts = [maxlen - n for n in lengths]
        cache.reset()
        return ids, mask, pos, lengths, pad_counts

    def _decode_step(
        self,
        cache: KVCache,
        sampler: Sampler,
        generated: list[list[int]],
        finished: list[bool],
        finish_reasons: list[str],
        cfg: GenerationConfig,
        pad_counts: list[int] | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """One sampling step. Returns ``(token_ids, produced_anything)``.

        ``pad_counts`` is the number of left-padding slots each row of the cache holds.
        The padding mask must be re-supplied on every decode step: dropping it would let a
        short row attend to the pad tokens cached during prefill, which silently changes
        its output as soon as it diverges from the batch.
        """
        next_ids = sampler.select(self._last_logits, generated)
        produced = False
        stop_ids = set(cfg.stop_token_ids)

        for b in range(len(generated)):
            if finished[b]:
                continue
            tok = int(next_ids[b])
            if tok in stop_ids:
                finished[b] = True
                finish_reasons[b] = "stop_token"
                continue
            generated[b].append(tok)
            produced = True

        if cfg.stop_strings:
            for b in range(len(generated)):
                if not finished[b]:
                    txt = self.tokenizer.decode(generated[b])
                    if any(s in txt for s in cfg.stop_strings):
                        finished[b] = True
                        finish_reasons[b] = "stop_string"

        ids = next_ids.unsqueeze(1)
        kv_len = cache.length + 1
        attn_mask = None
        if pad_counts and any(p > 0 for p in pad_counts):
            # One row of 1s per sequence, zeroed over each row's pad prefix.
            attn_mask = torch.ones((len(generated), kv_len), dtype=torch.bool, device=self.device)
            for b, p in enumerate(pad_counts):
                if p:
                    attn_mask[b, :p] = False
        out = self.model(
            ids,
            kv_cache=cache,
            kv_offset=cache.length,
            attention_mask=attn_mask,
            num_logits_to_keep=1,
        )
        self._last_logits = out.logits[:, -1, :].float()
        del out
        return next_ids, produced

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: Sequence[int] | Sequence[Sequence[int]],
        cfg: GenerationConfig,
        prefill_chunk: int = 512,
    ) -> list[GenerationResult]:
        """Generate completions for one or more prompts."""
        cache = self._ensure_cache(
            len(prompt_ids) if not (prompt_ids and isinstance(prompt_ids[0], int)) else 1,
            self._required_len(prompt_ids, cfg),
            self._kv_dtype(),
        )
        ids, mask, pos, lengths, pad_counts = self._prepare_batch(prompt_ids, cfg, cache)
        maxlen = ids.shape[1]
        B = len(lengths)

        sampler = Sampler(cfg, self.model.cfg.vocab_size, self.device)
        generated: list[list[int]] = [[] for _ in range(B)]
        finished = [False] * B
        finish_reasons = ["length"] * B

        t0 = time.perf_counter()
        self._prefill(ids, mask, pos, cache, maxlen, prefill_chunk)
        ttft = time.perf_counter() - t0
        synchronize(self.device)

        step = 0
        while step < cfg.max_new_tokens:
            _next, produced = self._decode_step(cache, sampler, generated, finished,
                                               finish_reasons, cfg, pad_counts)
            if not produced or all(finished):
                break
            if all(len(g) >= cfg.max_new_tokens for g in generated):
                break
            step += 1

        synchronize(self.device)
        elapsed = time.perf_counter() - t0
        return self._results(generated, finished, finish_reasons, lengths, elapsed, ttft, cfg)

    @torch.no_grad()
    def stream_ids(
        self,
        prompt_ids: Sequence[int] | Sequence[Sequence[int]],
        cfg: GenerationConfig,
        prefill_chunk: int = 512,
    ) -> Iterator[int]:
        """Yield generated token ids one at a time."""
        single = bool(prompt_ids) and isinstance(prompt_ids[0], int)
        batch = 1 if single else len(prompt_ids)  # type: ignore[arg-type]
        cache = self._ensure_cache(batch, self._required_len(prompt_ids, cfg), self._kv_dtype())
        ids, mask, pos, lengths, pad_counts = self._prepare_batch(prompt_ids, cfg, cache)
        maxlen = ids.shape[1]
        B = len(lengths)

        sampler = Sampler(cfg, self.model.cfg.vocab_size, self.device)
        generated: list[list[int]] = [[] for _ in range(B)]
        finished = [False] * B
        finish_reasons = ["length"] * B

        self._prefill(ids, mask, pos, cache, maxlen, prefill_chunk)
        synchronize(self.device)

        step = 0
        while step < cfg.max_new_tokens:
            before = [len(g) for g in generated]
            next_ids, produced = self._decode_step(cache, sampler, generated, finished,
                                                   finish_reasons, cfg, pad_counts)
            for b in range(B):
                if len(generated[b]) > before[b]:
                    yield generated[b][-1]
                    break
            if not produced or all(finished):
                break
            if all(len(g) >= cfg.max_new_tokens for g in generated):
                break
            step += 1

    def _required_len(self, prompt_ids: Any, cfg: GenerationConfig) -> int:
        if prompt_ids and isinstance(prompt_ids[0], int):
            longest = len(prompt_ids)
        else:
            longest = max((len(p) for p in prompt_ids), default=1)
        return longest + cfg.max_new_tokens + 8

    def _results(
        self,
        generated: list[list[int]],
        finished: list[bool],
        finish_reasons: list[str],
        lengths: list[int],
        elapsed: float,
        ttft: float,
        cfg: GenerationConfig,
    ) -> list[GenerationResult]:
        results = []
        for b in range(len(generated)):
            text = self.tokenizer.decode(generated[b])
            if cfg.stop_strings:
                for s in cfg.stop_strings:
                    if s in text:
                        text = text.split(s)[0]
                        break
            ct = len(generated[b])
            results.append(
                GenerationResult(
                    text=text,
                    token_ids=list(generated[b]),
                    prompt_tokens=lengths[b],
                    completion_tokens=ct,
                    finish_reason=finish_reasons[b] if finished[b] else "length",
                    seconds=elapsed,
                    tokens_per_second=ct / elapsed if elapsed > 0 else 0.0,
                    time_to_first_token=ttft,
                )
            )
        return results

    def _kv_dtype(self) -> torch.dtype:
        """Cache dtype must equal the model's compute dtype.

        Concatenating a cached fp16 key with a freshly computed fp32 key fails inside
        ``torch.cat``, and mixed-dtype caches also silently change attention numerics.
        """
        if self.dtype is not None:
            return self.dtype
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:  # pragma: no cover - model with no parameters
            return torch.float32

    def _prefill(self, ids, mask, pos, cache, maxlen, chunk) -> int:
        """Run prefill in chunks, leaving ``self._last_logits`` set for the next token.

        The padding mask passed to the model must span *every key seen so far*
        (``[0, end)``), not just the current chunk, otherwise chunks after the first would
        fail to mask the padding in earlier chunks. ``positions`` supplies each row's
        true RoPE offsets, which differ across a left-padded batch.
        """
        offset = 0
        for start in range(0, maxlen, chunk):
            end = min(maxlen, start + chunk)
            sub_ids = ids[:, start:end]
            sub_pos = pos[:, start:end]
            sub_mask = mask[:, :end] if mask is not None else None
            out = self.model(
                sub_ids,
                kv_cache=cache,
                kv_offset=offset,
                attention_mask=sub_mask,
                positions=sub_pos,
                num_logits_to_keep=1,
            )
            self._last_logits = out.logits[:, -1, :].float()
            offset = end
            del out
        return offset

    # Convenience -----------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        cfg: GenerationConfig | None = None,
        reset_cache: bool = True,
    ) -> GenerationResult:
        cfg = cfg or GenerationConfig()
        if reset_cache:
            self.reset_cache()
        ids = self.tokenizer.encode(prompt)
        if not ids:
            bos = getattr(self.tokenizer, "bos_id", None)
            ids = [bos if bos is not None else 0]
        results = self.generate_ids(ids, cfg)
        return results[0]

    def generate_chat(
        self,
        messages: Sequence[dict[str, str]],
        cfg: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = cfg or GenerationConfig()
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self.generate(prompt, cfg)

    def stream(
        self,
        prompt: str,
        cfg: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Yield decoded text deltas as they are produced."""
        cfg = cfg or GenerationConfig()
        self.reset_cache()
        ids = self.tokenizer.encode(prompt)
        if not ids:
            bos = getattr(self.tokenizer, "bos_id", None)
            ids = [bos if bos is not None else 0]
        for tid in self.stream_ids(ids, cfg):
            yield self.tokenizer.decode([tid])

    @torch.no_grad()
    def score(
        self,
        prompt: str,
        continuation: str,
    ) -> dict[str, float]:
        """Teacher-forced log-likelihood of ``continuation`` given ``prompt``.

        Used by the ranking stage of RAG and by preference evaluation. Returns the summed
        and mean log-probability so callers can compare candidates of different lengths
        without conflating length with quality.
        """
        p_ids = self.tokenizer.encode(prompt)
        c_ids = self.tokenizer.encode(continuation)
        if not c_ids:
            return {"sum_logprob": 0.0, "mean_logprob": 0.0, "n_tokens": 0, "perplexity": float("inf")}
        ids = p_ids + c_ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(input_ids).logits.float()
        lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        gathered = lp.gather(2, targets.unsqueeze(2)).squeeze(2)
        start = max(0, len(p_ids) - 1)
        relevant = gathered[0, start:]
        return {
            "sum_logprob": float(relevant.sum()),
            "mean_logprob": float(relevant.mean()),
            "n_tokens": int(relevant.numel()),
            "perplexity": float(torch.exp(-relevant.mean())),
        }


__all__ = ["Generator", "GenerationConfig", "GenerationResult", "Sampler"]
