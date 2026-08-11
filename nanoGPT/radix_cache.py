"""
PA1 Part C: reusing KV cache for shared prompts.

Builds a radix tree over a batch of tokenized prompts so that any prefix shared
by two or more prompts is prefilled through the transformer exactly once, and
its KV cache is reused (rather than recomputed) for every prompt/branch below
it. `generate_independent` is the baseline that processes each prompt from
scratch, with no cross-prompt reuse, for comparison.
"""

import torch
import torch.nn.functional as F


class RadixNode:
    def __init__(self, tokens):
        self.tokens = tokens          # this node's own token chunk (excludes ancestors)
        self.children = {}            # first token of child chunk -> RadixNode
        self.kv_cache = None          # per-block [(cache_k, cache_v), ...] after this chunk is processed
        self.end_prompts = []         # indices (into the input batch) of prompts ending exactly here


def _insert(root, tokens, prompt_idx):
    node = root
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok not in node.children:
            node.children[tok] = RadixNode(tokens[i:])
            node = node.children[tok]
            i = len(tokens)
            continue
        child = node.children[tok]
        j = 0
        while j < len(child.tokens) and i + j < len(tokens) and child.tokens[j] == tokens[i + j]:
            j += 1
        if j < len(child.tokens):
            # only a prefix of child.tokens is shared -> split the node
            split = RadixNode(child.tokens[:j])
            child.tokens = child.tokens[j:]
            split.children[child.tokens[0]] = child
            node.children[tok] = split
            node = split
        else:
            node = child
        i += j
    node.end_prompts.append(prompt_idx)


def build_radix_tree(prompts):
    """prompts: list of token-id lists. Returns the root RadixNode."""
    root = RadixNode([])
    for idx, toks in enumerate(prompts):
        assert len(toks) > 0, "empty prompts are not supported"
        _insert(root, toks, idx)
    return root


def _sample(logits, temperature, top_k, greedy):
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float('inf'))
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def _decode_continuation(model, first_logits, start_pos, max_new_tokens, temperature, top_k, greedy):
    """Continue generation given the model's cache already holding everything up to (not
    including) start_pos, and first_logits = the model's prediction for position start_pos
    (produced for free by the prefill/prefix forward pass, so no extra forward is spent on
    the first generated token)."""
    generated = []
    logits = first_logits
    curr_pos = start_pos
    for i in range(max_new_tokens):
        idx_next = _sample(logits, temperature, top_k, greedy)
        generated.append(idx_next.item())
        if i == max_new_tokens - 1:
            break
        out_logits, _ = model(idx_next, pos_offset=curr_pos)
        logits = out_logits[:, -1, :]
        curr_pos += 1
    return generated


@torch.no_grad()
def generate_radix_shared(model, prompts, max_new_tokens, device, temperature=1.0, top_k=None, greedy=False):
    """prompts: list of token-id lists (python ints). Returns list of generated continuations
    (one list of max_new_tokens token ids per prompt, same order as `prompts`)."""
    root = build_radix_tree(prompts)
    n_blocks = len(model.transformer.h)
    empty_cache = [(None, None)] * n_blocks
    results = [None] * len(prompts)

    def process(node, parent_cache, parent_len):
        if node.tokens:
            chunk = torch.tensor([node.tokens], dtype=torch.long, device=device)
            model.set_kv_cache(parent_cache)
            logits, _ = model(chunk, pos_offset=parent_len)
            node.kv_cache = model.get_kv_cache()
            node_len = parent_len + len(node.tokens)
            next_tok_logits = logits[:, -1, :]
        else:
            node.kv_cache = parent_cache
            node_len = parent_len
            next_tok_logits = None

        for pidx in node.end_prompts:
            model.set_kv_cache(node.kv_cache)
            results[pidx] = _decode_continuation(
                model, next_tok_logits, node_len, max_new_tokens, temperature, top_k, greedy
            )

        for child in node.children.values():
            process(child, node.kv_cache, node_len)

    process(root, empty_cache, 0)
    return results


@torch.no_grad()
def generate_independent(model, prompts, max_new_tokens, device, temperature=1.0, top_k=None, greedy=False):
    """Same KV-cache machinery as generate_radix_shared, but each prompt is prefilled and
    decoded on its own -- no cross-prompt reuse, even when prompts share a prefix."""
    results = []
    for toks in prompts:
        idx = torch.tensor([toks], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k,
                              use_cache=True, greedy=greedy)
        results.append(out[0, len(toks):].tolist())
    return results
