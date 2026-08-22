"""The tensor half of continuous batching.

Implements `BatchedModel` for HuggingFace transformers. The scheduler decides
who runs; this decides how.

Two things make it work:

* **Left padding.** Generation always reads the last position, so sequences of
  different lengths are aligned at the right edge and the padding sits in front,
  masked out. Right padding would make the model attend to pad tokens as if they
  were the newest input.
* **One persistent batched cache.** A batched `DynamicCache` is one tensor per
  layer with the batch on dimension 0. Surgery happens only when the batch
  *changes*: joining is a pad-and-concatenate, leaving is an index-select. A
  first draft split and re-merged the cache on every step, which is O(cache) of
  memory traffic per token and would have been slower than not batching at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SeqState:
    """Per-sequence tensor state, hung off `Sequence.handle`."""

    input_ids: Any = None       # [1, prompt_len]
    cache_len: int = 0
    last_token: Any = None      # [1, 1]
    banned: list[list[int]] = field(default_factory=list)
    # Decoding must span tokens: Qwen uses byte-level BPE, so one emoji is
    # several tokens and decoding them individually yields U+FFFD. The ids are
    # kept and re-decoded so only completed characters are emitted.
    token_ids: list[int] = field(default_factory=list)
    emitted: str = ""

    def take(self, tokenizer, token_id: int) -> str:
        """Append a token and return whatever text became complete."""
        self.token_ids.append(int(token_id))
        text = tokenizer.decode(self.token_ids, skip_special_tokens=True)
        # A half-finished character decodes to the replacement char; hold it
        # back until the following token completes it.
        if text.endswith("\ufffd"):
            return ""
        piece = text[len(self.emitted) :]
        self.emitted = text
        return piece


class TorchBatchedModel:
    def __init__(self, tokenizer, model, *, torch_module, banned_ids=None) -> None:
        self.tok = tokenizer
        self.model = model
        self.torch = torch_module
        self._banned = banned_ids or []
        self._cache = None
        self._order: list[str] = []
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id

    # --- helpers -----------------------------------------------------------

    def _render(self, sequence) -> Any:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if sequence.tools:
            kwargs["tools"] = sequence.tools
        try:
            prompt = self.tok.apply_chat_template(sequence.messages, **kwargs)
        except TypeError:
            log.warning("chat template without tool support; continuing without")
            prompt = self.tok.apply_chat_template(
                sequence.messages, tokenize=False, add_generation_prompt=True
            )
        return self.tok(prompt, return_tensors="pt").input_ids.to(self.model.device)

    def _mask_for(self, sequences) -> Any:
        """Left-padded attention mask covering cache plus the new token."""
        torch = self.torch
        longest = max(s.handle.cache_len for s in sequences)
        rows = []
        for s in sequences:
            pad = longest - s.handle.cache_len
            rows.append(
                torch.cat(
                    [
                        torch.zeros(pad, dtype=torch.long),
                        torch.ones(s.handle.cache_len + 1, dtype=torch.long),
                    ]
                )
            )
        return torch.stack(rows).to(self.model.device)

    # --- BatchedModel ------------------------------------------------------

    def prefill(self, sequences) -> None:
        torch = self.torch
        for sequence in sequences:
            ids = self._render(sequence)
            state = SeqState(input_ids=ids, banned=self._banned)
            with torch.inference_mode():
                out = self.model(input_ids=ids, use_cache=True)
            state.cache_len = int(ids.shape[1])
            state.last_token = self._sample(out.logits[:, -1, :], sequence, state)
            sequence.handle = state
            # Each sequence keeps its own cache until it joins the batch; the
            # first decode step merges them.
            state.cache = out.past_key_values

    def _sample(self, logits, sequence, state) -> Any:
        torch = self.torch
        logits = logits.clone()
        for ids in state.banned:
            if len(ids) == 1:
                logits[:, ids[0]] = float("-inf")
        if sequence.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.softmax(logits / max(0.01, sequence.temperature), dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def decode(self, sequences) -> list[tuple[Any, str, bool]]:
        torch = self.torch
        eos = self.tok.eos_token_id
        self._resync(sequences)

        tokens = torch.cat([s.handle.last_token for s in sequences], dim=0)
        with torch.inference_mode():
            out = self.model(
                input_ids=tokens,
                past_key_values=self._cache,
                attention_mask=self._mask_for(sequences),
                use_cache=True,
            )
        # The model grew the cache in place; keep the handle to it.
        self._cache = out.past_key_values

        results: list[tuple[Any, str, bool]] = []
        for index, sequence in enumerate(sequences):
            state = sequence.handle
            piece = state.take(self.tok, state.last_token[0, 0].item())
            nxt = self._sample(out.logits[index : index + 1, -1, :], sequence, state)
            state.last_token = nxt
            state.cache_len += 1
            done = eos is not None and int(nxt.item()) == int(eos)
            results.append((sequence, piece, done))
        return results

    def _resync(self, sequences) -> None:
        """Bring the batched cache in line with who is actually decoding."""
        wanted = [s.id for s in sequences]
        if wanted == self._order and self._cache is not None:
            return

        keep = [i for i, sid in enumerate(self._order) if sid in set(wanted)]
        base = _select(self._cache, keep, self.torch) if (self._cache is not None and keep) else None
        kept_ids = [self._order[i] for i in keep]

        joining = [s for s in sequences if s.id not in set(kept_ids)]
        parts = [base] if base is not None else []
        parts.extend(s.handle.cache for s in joining)
        parts = [p for p in parts if p is not None]

        self._cache = _merge_caches(parts, self.torch) if len(parts) > 1 else (parts[0] if parts else None)
        self._order = kept_ids + [s.id for s in joining]
        for sequence in joining:
            # The private cache has been folded into the batch.
            sequence.handle.cache = None

        # decode() indexes by position, so the caller's order must match.
        if self._order != wanted:
            order = _select(self._cache, [self._order.index(sid) for sid in wanted], self.torch)
            self._cache = order
            self._order = list(wanted)

    def release(self, sequences) -> None:
        gone = {s.id for s in sequences}
        for sequence in sequences:
            if sequence.handle is not None:
                sequence.handle.cache = None
                sequence.handle = None
        keep = [i for i, sid in enumerate(self._order) if sid not in gone]
        if self._cache is not None and keep:
            self._cache = _select(self._cache, keep, self.torch)
            self._order = [self._order[i] for i in keep]
        else:
            self._cache = None
            self._order = []
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _select(cache, indices, torch):
    """Keep only these batch rows of a batched cache."""
    from transformers import DynamicCache

    out = DynamicCache()
    picker = torch.tensor(indices, device=cache[0][0].device)
    for layer in range(len(cache)):
        k, v = cache[layer]
        out.update(k.index_select(0, picker), v.index_select(0, picker), layer)
    return out


def _merge_caches(caches, torch):
    """Left-pad each cache to the longest and stack them on the batch axis."""
    if len(caches) == 1:
        return caches[0]
    from transformers import DynamicCache

    longest = max(c.get_seq_length() for c in caches)
    merged = DynamicCache()
    layers = len(caches[0])
    for layer in range(layers):
        keys, values = [], []
        for cache in caches:
            k, v = cache[layer]
            pad = longest - k.shape[2]
            if pad > 0:
                shape = (k.shape[0], k.shape[1], pad, k.shape[3])
                zeros_k = torch.zeros(shape, dtype=k.dtype, device=k.device)
                zeros_v = torch.zeros(shape, dtype=v.dtype, device=v.device)
                k = torch.cat([zeros_k, k], dim=2)
                v = torch.cat([zeros_v, v], dim=2)
            keys.append(k)
            values.append(v)
        merged.update(torch.cat(keys, dim=0), torch.cat(values, dim=0), layer)
    return merged
