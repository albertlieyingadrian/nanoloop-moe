"""
nanoloop-moe: a small, readable looped mixture-of-experts transformer in the style of
Inkling 2 Mini (an Inkling-style toy; not affiliated with Thinking Machines).

Three ideas on top of a plain GPT, each one a few dozen lines below:

1. A LOOP. The layers are split into a prelude (runs once), a core (runs K times with the
   same weights) and a coda (runs once). K is chosen at inference.

2. NO SINGLE RESIDUAL STREAM. Layers are grouped into blocks. Sublayer outputs only add up
   inside the block being built (the "active block"). When a block ends it is frozen into a
   list of finished blocks, and every sublayer reads its OWN learned mix of that list plus
   the active block (depth attention, "attention residuals"). The output head reads one more
   mix, `final_input`.

3. A GATED MESSAGE BETWEEN PASSES. Every core pass restarts from the prelude's blocks.
   What the previous pass worked out reaches the next one only as a message: learned queries
   read the previous pass's core blocks, and a low-rank tanh gate merges the message into
   the attention and MoE inputs, x <- (1 - g_forget) * x + g_message * message.

The feed-forward sublayer is a top-k mixture of experts with a sigmoid router.

    tokens -> embed -> [prelude] -> [core] x K -> [coda] -> final_input mix -> logits
                          |           ^   |
                          |           +---+ message bank (previous pass's core blocks)
                          +-- every core pass restarts from the prelude's blocks

One file, PyTorch only (matplotlib for `--mode plot`):

    python nanoloop_moe.py --mode compare                 # chain task: 5 models at matched active params
    python nanoloop_moe.py --mode compare --task shakespeare --steps 3000
    python nanoloop_moe.py --mode sample --out out/loopmoe.pt --loops 1   # K at inference
    python nanoloop_moe.py --mode lens --out out/loopmoe.pt                # logit lens vs mix lens
"""
import argparse
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import asdict, dataclass, fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    task: str = "chain"          # "chain" (multi-hop lookup) or "shakespeare" (char LM)
    arch: str = "loopmoe"        # "loopmoe" (this model), "plainloop" (one residual stream) or "standard"
    n_embd: int = 128
    n_head: int = 4
    block_layers: int = 2        # layers per block (Inkling 2 Mini: 6)
    prelude_blocks: int = 1      # Inkling 2 Mini: 1 block = layers 0-5
    core_blocks: int = 2         # Inkling 2 Mini: 2 blocks = layers 6-17, looped
    coda_blocks: int = 1         # Inkling 2 Mini: 1 block = layers 18-23
    loops: int = 3               # K: core passes (Inkling 2 Mini: 3, fixed)
    random_loops: bool = False   # train with K ~ Uniform{1..loops} (like the variable-loop research model)
    mix: bool = True             # block mixing; False = read the plain sum of blocks (a residual stream)
    messages: bool = True        # gated messages between core passes
    gate_rank: int = 16          # rank of the message gates (Inkling 2 Mini: 64)
    moe: bool = True             # top-k MoE; False = a dense SwiGLU with the same active width
    n_experts: int = 16          # Inkling 2 Mini: 128
    top_k: int = 4               # Inkling 2 Mini: 8
    n_shared: int = 0            # always-on shared experts (Inkling 2 Mini: 0)
    d_expert: int = 64           # expert width (Inkling 2 Mini: hidden/2 = 1152)
    bias_rate: float = 1e-3      # load balancing: router-bias nudge per step (0 = off)
    hops: int = 8                # chain task: max lookups needed
    block_size: int = 96         # context length in characters
    batch_size: int = 128
    steps: int = 6000
    lr: float = 5e-4
    seed: int = 1337
    device: str = "auto"         # auto picks cuda, then mps, then cpu
    out: str = "out/model.pt"
    vocab: str = ""              # filled in from the data; saved with the checkpoint

    @property
    def n_layer(self):
        return self.block_layers * (self.prelude_blocks + self.core_blocks + self.coda_blocks)


# ----------------------------------------------------------------------------- building blocks

def rms(x, eps=1e-6):
    """RMS norm with no learned scale: every vector is rescaled to length sqrt(dim)."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rms(x) * self.weight


def rope(x):
    """Rotary positions: rotate (even, odd) feature pairs by an angle proportional to position,
    so q.k depends on the distance between tokens. (A stand-in: Inkling uses a learned relative
    position bias instead; see the README.)"""
    T, D = x.shape[-2], x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device) / D))
    ang = torch.arange(T, device=x.device)[:, None] * freqs
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)


class Attention(nn.Module):
    """Causal multi-head attention. Queries, keys and values may come from DIFFERENT inputs,
    because in a block-mixing model each of them reads its own mix of the blocks."""

    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.q, self.k, self.v, self.o = (nn.Linear(c.n_embd, c.n_embd, bias=False) for _ in range(4))

    def forward(self, xq, xk, xv):
        B, T, C = xq.shape
        q, k, v = (f(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for f, x in ((self.q, xq), (self.k, xk), (self.v, xv)))
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


def softcap(x, cap):
    return torch.tanh(x / cap) * cap


class SwiGLU(nn.Module):
    """One expert (or a dense FFN): down(softcap(g) * sigmoid(g) * softcap(up)), as in Inkling's
    softcapped SwiGLU (the sigmoid sees the raw gate; the caps only bound the linear factors)."""

    def __init__(self, d, hidden):
        super().__init__()
        self.gate, self.up = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        g = self.gate(x)
        return self.down(softcap(g, 4.0) * torch.sigmoid(g) * softcap(self.up(x), 25.0))


class MoE(nn.Module):
    """Top-k mixture of experts with a sigmoid router.

    Each token scores every expert with sigmoid(router(x)), keeps the top_k, renormalizes their
    scores to sum to 1, and scales them by `route_scale` (= top_k here). `bias` is added to the
    scores ONLY when choosing experts, never when weighting them: it is the load-balancing knob.
    After every training step `balance()` raises the bias of under-used experts and lowers it for
    over-used ones (auxiliary-loss-free balancing). Optional shared experts see every token.

    The experts' weights are stacked into (E, ...) tensors. At toy size it is faster to run every
    expert on every token and multiply the unchosen ones by a weight of exactly 0 than to gather
    tokens per expert; the result is the same. Only top_k experts count as "active" parameters."""

    def __init__(self, c):
        super().__init__()
        E, D, H = c.n_experts, c.n_embd, c.d_expert
        self.top_k, self.route_scale = c.top_k, float(c.top_k)
        self.router = nn.Linear(D, E, bias=False)
        self.w_gate, self.w_up = (nn.Parameter(torch.randn(E, D, H) / math.sqrt(D)) for _ in range(2))
        self.w_down = nn.Parameter(torch.randn(E, H, D) / math.sqrt(H))
        self.shared = SwiGLU(D, c.n_shared * H) if c.n_shared else None
        self.register_buffer("bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))  # tokens per expert since the last balance()

    def route(self, flat):
        """(N, E) routing weights: top_k nonzero entries per token, summing to route_scale."""
        scores = torch.sigmoid(self.router(flat).float())
        idx = torch.topk(scores + self.bias, self.top_k, dim=-1).indices   # (N, k): which experts
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.route_scale
        return torch.zeros_like(scores).scatter(1, idx, w).to(flat.dtype), idx

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        w, idx = self.route(flat)
        if self.training:
            self.load += torch.bincount(idx.flatten(), minlength=len(self.bias)).float()
        g = torch.einsum("nd,edh->neh", flat, self.w_gate)
        hid = softcap(g, 4.0) * torch.sigmoid(g) * softcap(torch.einsum("nd,edh->neh", flat, self.w_up), 25.0)
        out = torch.einsum("neh,ehd,ne->nd", hid, self.w_down, w)  # weighted sum over experts
        if self.shared is not None:
            out = out + self.shared(flat)
        return out.view_as(x)

    @torch.no_grad()
    def balance(self, rate):
        if self.load.sum() > 0:
            self.bias += rate * torch.sign(self.load.mean() - self.load)
        self.load.zero_()


def ffn(c):
    if c.moe:
        return MoE(c)
    return SwiGLU(c.n_embd, (c.top_k + c.n_shared) * c.d_expert)  # same active width as the MoE


# ----------------------------------------------------------------------------- block mixing and messages

def depth_attention(query, stack):
    """Softmax over the block axis of (normalized block . query): the mix weights.
    stack: (B, T, N, D) already normalized; query: (Q, D). Returns (B, T, N, Q)."""
    return torch.softmax(stack @ query.float().T / math.sqrt(stack.shape[-1]), dim=-2)


class DepthMix(nn.Module):
    """Depth attention over blocks: the input a sublayer reads.

    Stack the finished blocks and the active block, RMS-normalize each, score each against a
    learned query vector, softmax over blocks, and return the weighted average of the normalized
    blocks. `n_query` queries give `n_query` different mixes (one each for q, k, v).
    Queries start at zero, so every mix starts as a plain average.

    With mix=False this returns the plain SUM of the blocks instead: that sum is exactly the
    residual stream of an ordinary transformer, which is what makes it a clean ablation."""

    def __init__(self, c, n_query=1):
        super().__init__()
        self.mix = c.mix
        self.query = nn.Parameter(torch.zeros(n_query, c.n_embd))

    def weights(self, blocks, active):
        normed = rms(torch.cat([blocks, active.unsqueeze(-2)], dim=-2).float())   # (B, T, N+1, D)
        return normed, depth_attention(self.query, normed)

    def forward(self, blocks, active):
        if not self.mix:
            return [blocks.sum(-2) + active] * self.query.shape[0]
        normed, w = self.weights(blocks, active)
        return list(torch.einsum("btnq,btnd->btqd", w, normed).to(active.dtype).unbind(-2))


class MessageGate(nn.Module):
    """x <- (1 - g_forget) * x + g_message * message, with both gates tanh of a rank-r projection
    of [norm(x), norm(message)]. Both projections start at zero, so at the start of training the
    gate blocks the whole message and every pass behaves like a non-looped model."""

    def __init__(self, c):
        super().__init__()
        self.down = nn.Linear(2 * c.n_embd, c.gate_rank, bias=False)
        self.forget = nn.Linear(c.gate_rank, c.n_embd, bias=False)
        self.message = nn.Linear(c.gate_rank, c.n_embd, bias=False)
        nn.init.zeros_(self.forget.weight)
        nn.init.zeros_(self.message.weight)

    def forward(self, x, msg):
        z = self.down(torch.cat([rms(x), rms(msg)], dim=-1))
        return (1 - torch.tanh(self.forget(z))) * x + torch.tanh(self.message(z)) * msg


class MessageInput(nn.Module):
    """Reads four messages (for q, k, v and the MoE input) per core layer from the message bank:
    the previous pass's finished core blocks plus its last active block. One learned query per
    message, the same depth attention as DepthMix."""

    def __init__(self, c, n_core_layers):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_core_layers, 4, c.n_embd))

    def forward(self, bank_blocks, bank_active, core_layer):
        stack = rms(torch.cat([bank_blocks, bank_active.unsqueeze(-2)], dim=-2).float())
        w = depth_attention(self.query[core_layer], stack)
        return list(torch.einsum("btnq,btnd->btqd", w, stack).to(bank_active.dtype).unbind(-2))


class Layer(nn.Module):
    """One decoder layer. `h` is the active block (the sum of sublayer outputs since the block
    started); `blocks` is (B, T, N, D), the finished blocks so far."""

    def __init__(self, c, idx, looped):
        super().__init__()
        self.boundary = idx % c.block_layers == 0  # this layer starts a new block
        self.attn_input = DepthMix(c, n_query=3) if idx > 0 else None
        self.mlp_input = DepthMix(c)
        self.norm_q, self.norm_k, self.norm_v, self.norm_mlp = (RMSNorm(c.n_embd) for _ in range(4))
        self.attn, self.ffn = Attention(c), ffn(c)
        self.gates = nn.ModuleList(MessageGate(c) for _ in range(4)) if looped and c.messages else None

    def forward(self, h, blocks, msgs=None, rec=None):
        start = h
        xq, xk, xv = self.attn_input(blocks, h) if self.attn_input else (h, h, h)
        if msgs is not None and self.gates is not None:
            xq, xk, xv = (g(x, m) for g, x, m in zip(self.gates[:3], (xq, xk, xv), msgs[:3]))
        a = self.attn(self.norm_q(xq), self.norm_k(xk), self.norm_v(xv))
        if self.boundary:  # freeze the block we were building; start a new one with this output
            blocks = torch.cat([blocks, start.unsqueeze(-2)], dim=-2)
            h = a
        else:
            h = start + a
        x = self.mlp_input(blocks, h)[0]
        if msgs is not None and self.gates is not None:
            x = self.gates[3](x, msgs[3])
        if rec is not None:
            rec.append({"act_in": start, "mlp_in": x, "blocks": blocks})
        return h + self.ffn(self.norm_mlp(x)), blocks


class LoopMoE(nn.Module):
    """Prelude -> core x K (each pass restarts from the prelude's blocks, plus a message) -> coda."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        L, p, k = c.block_layers, c.prelude_blocks, c.core_blocks
        self.core_start, self.core_end = L * p, L * (p + k)
        self.tok = nn.Embedding(len(c.vocab), c.n_embd)
        self.layers = nn.ModuleList(Layer(c, i, self.core_start <= i < self.core_end) for i in range(c.n_layer))
        self.message_input = MessageInput(c, self.core_end - self.core_start)
        self.final_input = DepthMix(c)
        self.norm = RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, len(c.vocab), bias=False)

    def visits(self, K):
        """(layer index, pass) for every layer visit, in the order they run."""
        core = range(self.core_start, self.core_end)
        return ([(i, 0) for i in range(self.core_start)] + [(i, p) for p in range(K) for i in core]
                + [(i, 0) for i in range(self.core_end, self.c.n_layer)])

    def forward(self, idx, loops=None, targets=None, rec=None):
        K = loops or self.c.loops
        h = self.tok(idx)
        blocks = h.new_zeros(*h.shape[:-1], 0, h.shape[-1])
        for layer in self.layers[:self.core_start]:
            h, blocks = layer(h, blocks, rec=rec)
        pre_h, pre_blocks = h, blocks
        n_pre, bank = blocks.shape[-2], None
        for _ in range(K):
            h, blocks = pre_h, pre_blocks  # every pass restarts from the prelude
            for j, layer in enumerate(self.layers[self.core_start:self.core_end]):
                msgs = self.message_input(*bank, j) if bank is not None and self.c.messages else None
                h, blocks = layer(h, blocks, msgs, rec=rec)
            # +1 skips the prelude's own block, which the first core layer froze into the list
            bank = (blocks[..., n_pre + 1:, :], h)
        for layer in self.layers[self.core_end:]:
            h, blocks = layer(h, blocks, rec=rec)
        if rec is not None:
            rec.append({"act_in": h, "blocks": blocks})
        return self.output(self.final_input(blocks, h)[0], targets)

    def output(self, x, targets):
        logits = softcap(self.head(self.norm(x)), 30.0)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


class PlainLayer(nn.Module):
    """Baseline layer with one residual stream: x = x + attn(norm(x)); x = x + ffn(norm(x))."""

    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.attn, self.ffn = Attention(c), ffn(c)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n, n)
        return x + self.ffn(self.n2(x))


class PlainLoop(LoopMoE):
    """The nanoloop-transformer recipe at the same layout: one residual stream, the core looped
    K times with input injection (re-add the prelude output e before every pass).
    arch="standard" is the same network trained with K=1: an ordinary transformer."""

    def __init__(self, c):
        nn.Module.__init__(self)
        self.c = c
        self.core_start = c.block_layers * c.prelude_blocks
        self.core_end = self.core_start + c.block_layers * c.core_blocks
        self.tok = nn.Embedding(len(c.vocab), c.n_embd)
        self.layers = nn.ModuleList(PlainLayer(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, len(c.vocab), bias=False)

    def forward(self, idx, loops=None, targets=None, rec=None):
        K = loops or self.c.loops
        x = self.tok(idx)
        for layer in self.layers[:self.core_start]:
            x = layer(x)
        e, x = x, torch.zeros_like(x)
        for _ in range(K):
            x = x + e
            for layer in self.layers[self.core_start:self.core_end]:
                if rec is not None:
                    rec.append({"act_in": x})
                x = layer(x)
        for layer in self.layers[self.core_end:]:
            if rec is not None:
                rec.append({"act_in": x})
            x = layer(x)
        if rec is not None:
            rec.append({"act_in": x})
        return self.output(x, targets)


def build(c):
    return LoopMoE(c) if c.arch == "loopmoe" else PlainLoop(c)


def count_params(model):
    """(stored, active per token). Active counts only the top_k routed experts (plus shared)."""
    total, idle = sum(p.numel() for p in model.parameters()), 0
    for m in model.modules():
        if isinstance(m, MoE):
            per_expert = (m.w_gate.numel() + m.w_up.numel() + m.w_down.numel()) // len(m.bias)
            idle += per_expert * (len(m.bias) - m.top_k)
    return total, total - idle


# ----------------------------------------------------------------------------- data

CHAIN_VOCAB = ".abcdefghijklmnopqrstuvwxyz0123456789=;?"  # "." is padding


def chain_example(rng, max_hops, n_chains=2):
    """A pointer-chasing puzzle (from nanoloop-transformer): two chains like a=7;b=a;c=b,
    interleaved and written defined-before-use, then one question per variable.
    'c?' needs 3 lookups in a row (c->b->a->7). Returns the text and
    [(index of '?', answer digit, hops needed)]."""
    lens = [rng.randint(1, max_hops) for _ in range(n_chains)]
    names = iter(rng.sample("abcdefghijklmnopqrstuvwxyz", sum(lens)))
    chains, answers = [], {}
    for n in lens:
        vs, digit = [next(names) for _ in range(n)], str(rng.randrange(10))
        for i, v in enumerate(vs):
            answers[v] = (digit, n - i)
        chains.append([f"{v}={t}" for v, t in reversed(list(zip(vs, vs[1:] + [digit])))])
    statements = []
    while any(chains):
        statements.append(rng.choice([ch for ch in chains if ch]).pop(0))
    queries = list(answers)
    rng.shuffle(queries)
    text = ";".join(statements) + ";" + "".join(f"{q}?" for q in queries)
    start = len(text) - 2 * len(queries)
    return text, [(start + 2 * i + 1, *answers[q]) for i, q in enumerate(queries)]


def chain_batch(rng, c, n=None):
    """x: puzzles left-padded with '.'; y: the answer digit at every '?', -1 elsewhere;
    hops: lookups needed at each '?', 0 elsewhere."""
    n = n or c.batch_size
    stoi = {ch: i for i, ch in enumerate(CHAIN_VOCAB)}
    x = torch.zeros(n, c.block_size, dtype=torch.long)
    y = torch.full((n, c.block_size), -1, dtype=torch.long)
    hops = torch.zeros(n, c.block_size, dtype=torch.long)
    for b in range(n):
        text, qs = chain_example(rng, c.hops)
        off = c.block_size - len(text)
        x[b, off:] = torch.tensor([stoi[ch] for ch in text])
        for i, digit, h in qs:
            y[b, off + i], hops[b, off + i] = stoi[digit], h
    return x, y, hops


def load_text():
    """Tiny Shakespeare as char ids, split 90/10 into train and validation."""
    path = "data/tinyshakespeare.txt"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    text = open(path).read()
    vocab = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text])
    split = int(0.9 * len(data))
    return vocab, data[:split], data[split:]


def text_batch(data, rng, c):
    ix = [rng.randrange(len(data) - c.block_size - 1) for _ in range(c.batch_size)]
    return (torch.stack([data[i:i + c.block_size] for i in ix]),
            torch.stack([data[i + 1:i + c.block_size + 1] for i in ix]))


# ----------------------------------------------------------------------------- train / eval

def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def chain_accuracy(model, c, loops, n=1000, seed=0):
    """Accuracy per hop count on a fixed held-out set (same seed -> same puzzles)."""
    model.eval()
    dev = next(model.parameters()).device
    x, y, hops = chain_batch(random.Random(seed), c, n=n)
    pred = torch.cat([model(xb.to(dev), loops)[0].argmax(-1).cpu() for xb in x.split(250)])
    model.train()
    correct = pred == y
    return {h: correct[hops == h].float().mean().item() for h in range(1, c.hops + 1)}


@torch.no_grad()
def val_loss(model, data, c, loops, n=10):
    model.eval()
    rng, dev = random.Random(0), next(model.parameters()).device
    batches = [text_batch(data, rng, c) for _ in range(n)]
    losses = [model(x.to(dev), loops, y.to(dev))[1].item() for x, y in batches]
    model.train()
    return sum(losses) / n


def train(c, log=print):
    torch.manual_seed(c.seed)
    rng, dev = random.Random(c.seed), pick_device(c.device)
    level, acc = 1, 0.0
    if c.task == "chain":
        c.vocab = CHAIN_VOCAB
        get_batch = lambda: chain_batch(rng, replace(c, hops=level))[:2]  # noqa: E731
    else:
        c.vocab, train_data, val_data = load_text()
        get_batch = lambda: text_batch(train_data, rng, c)  # noqa: E731
    model = build(c).to(dev)
    total, active = count_params(model)
    log(f"{c.arch}: {total:,} params, {active:,} active | K={c.loops} mix={c.mix} msgs={c.messages} "
        f"moe={c.moe} | {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=0.0)
    moes = [m for m in model.modules() if isinstance(m, MoE)]
    t0, history = time.time(), []
    for step in range(1, c.steps + 1):
        x, y = get_batch()
        K = rng.randint(1, c.loops) if c.random_loops else c.loops
        x, y = x.to(dev), y.to(dev)
        logits, loss = model(x, K, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for m in moes:
            m.balance(c.bias_rate)
        if c.task == "chain":  # curriculum: unlock one more hop when train accuracy passes 85%
            scored = y >= 0
            acc = 0.95 * acc + 0.05 * (logits.argmax(-1)[scored] == y[scored]).float().mean().item()
            if acc > 0.85 and level < c.hops:
                level, acc = level + 1, 0.0
                log(f"step {step:5d} | curriculum: now up to {level} hops")
        if step % max(1, c.steps // 10) == 0 or step == 1:
            msg = f"step {step:5d} | loss {loss.item():.3f}"
            if c.task == "chain":
                msg += f" | train acc {acc:.2f} at <={level} hops"
            else:
                vl = val_loss(model, val_data, c, c.loops)
                history.append((step, vl))
                msg += f" | val {vl:.3f}"
            log(msg + f" | {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(c.out) or ".", exist_ok=True)
    torch.save({"config": asdict(c), "model": model.state_dict(), "level": level, "history": history}, c.out)
    return model


def load(path, device="auto"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    c = Config(**ckpt["config"])
    model = build(c)
    model.load_state_dict(ckpt["model"])
    return model.to(pick_device(device)).eval(), c


def sweep(model, c, max_loops=6):
    """Chain accuracy for K = 1..max_loops: {K: {hop: acc}}."""
    return {K: chain_accuracy(model, c, K) for K in range(1, max_loops + 1)}


def compare_configs(base):
    """Five models at matched ACTIVE parameters per token (same width, same layer layout)."""
    def o(name, **kw):
        return name, replace(base, out=f"out/{base.task}_{name}.pt", **kw)
    return [
        o("standard", arch="standard", moe=False, loops=1),  # one stream, no loop, dense FFN
        o("plainloop", arch="plainloop", moe=False),         # the nanoloop-transformer recipe
        o("loopmoe_k1", loops=1),                            # this model without the loop
        o("loopmoe_nomix", mix=False),                       # loop + messages, plain sum instead of mixing
        o("loopmoe"),                                        # the full model
    ]


def compare(base, only=None):
    path = f"out/compare_{base.task}.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    for name, c in compare_configs(base):
        if only and name not in only:
            continue
        t0 = time.time()
        model = train(c, log=lambda msg, name=name: print(f"[{name}] {msg}", flush=True))
        results[name] = {**score(model, c), "seconds": round(time.time() - t0)}
        os.makedirs("out", exist_ok=True)
        json.dump(results, open(path, "w"), indent=1)
    print_table(results)
    return results


def score(model, c):
    """The numbers behind a results row: parameter counts, then the K sweep (chain) or val loss per K (text)."""
    total, active = count_params(model)
    row = {"params": total, "active": active, "steps": c.steps}
    if c.task == "chain":
        row["sweep"] = sweep(model, c)
        row["acc"] = row["sweep"][c.loops]
        row["mean"] = sum(row["acc"].values()) / len(row["acc"])
    else:
        _, _, val_data = load_text()
        row["val"] = {K: val_loss(model, val_data, c, K) for K in (1, 2, 3, 4, 6)}
    return row


def print_table(results):
    for name, r in results.items():
        if "acc" in r:
            print(f"{name:14s} {r['params']:>9,} {r['active']:>9,}  mean {r['mean']:.3f}  "
                  + " ".join(f"{h}:{a:.2f}" for h, a in r["acc"].items()))
        else:
            print(f"{name:14s} {r['params']:>9,} {r['active']:>9,}  val "
                  + " ".join(f"K{K}:{v:.3f}" for K, v in r["val"].items()))


# ----------------------------------------------------------------------------- lens

@torch.no_grad()
def lens(model, c, loops=None, n=500, seed=0):
    """Read every layer visit at the '?' positions; score whether the top token is the answer.

    loopmoe has no single residual stream, so there are three things to read at a visit:
      act_in   the active block entering the visit, unembedded (the plain logit lens)
      mlp_in   the visit's own MoE-input mix of the blocks, message gated in
      mix      the output head's final_input mix over the current block list + active block:
               what the model would output if it stopped here (the "mix lens")
    plainloop/standard: the one residual stream entering each visit (the plain logit lens).
    The last entry is the model's real output. Returns {lens: [accuracy per visit], "visits": labels}."""
    model.eval()
    K = loops or c.loops
    dev = next(model.parameters()).device
    x, y, _ = chain_batch(random.Random(seed), c, n=n)
    mask, rec = y >= 0, []
    model(x.to(dev), K, rec=rec)
    gold = y[mask]

    def read(s):
        return (model.head(model.norm(s)).argmax(-1).cpu()[mask] == gold).float().mean().item()

    if c.arch != "loopmoe":
        labels = [f"L{i} p{p + 1}" for p in range(K) for i in range(model.core_start, model.core_end)]
        labels += [f"L{i}" for i in range(model.core_end, c.n_layer)] + ["out"]
        return {"visits": labels, "act_in": [read(r["act_in"]) for r in rec]}
    out = {"visits": [], "act_in": [], "mlp_in": [], "mix": []}
    looped = range(model.core_start, model.core_end)
    for (layer, p), r in zip(model.visits(K) + [("out", 0)], rec):
        out["visits"].append("out" if layer == "out" else f"L{layer} p{p + 1}" if layer in looped else f"L{layer}")
        out["act_in"].append(read(r["act_in"]))
        out["mlp_in"].append(read(r["mlp_in"]) if "mlp_in" in r else float("nan"))
        out["mix"].append(read(model.final_input(r["blocks"], r["act_in"])[0]))
    _, w = model.final_input.weights(rec[-1]["blocks"], rec[-1]["act_in"])
    out["final_weights"] = [round(v, 3) for v in w[mask.to(dev)].mean(0)[:, 0].tolist()]
    return out


def print_lens(out):
    keys = [k for k in ("act_in", "mlp_in", "mix") if k in out]
    print("visit      " + "  ".join(f"{k:>7s}" for k in keys))
    for i, v in enumerate(out["visits"]):
        print(f"{v:10s} " + "  ".join(f"{out[k][i]:7.3f}" for k in keys))
    if "final_weights" in out:
        print("final_input weights at the end over [embed, blocks..., active]:", out["final_weights"])


# ----------------------------------------------------------------------------- plots

def plot(suffix="", path="assets"):
    """assets/k_sweep.png from out/compare_chain<suffix>.json and assets/lens.png from out/lens<suffix>.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(path, exist_ok=True)
    if os.path.exists(f := f"out/compare_chain{suffix}.json"):
        res = json.load(open(f))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for name, r in res.items():
            Ks = sorted(r["sweep"], key=int)
            ax.plot([int(k) for k in Ks], [sum(r["sweep"][k].values()) / len(r["sweep"][k]) for k in Ks],
                    marker="o", label=name)
        ax.set(xlabel="core passes K at inference", ylabel="chain accuracy (mean over 1-8 hops)", ylim=(0, 1.02))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{path}/k_sweep.png", dpi=150)
    if os.path.exists(f := f"out/lens{suffix}.json"):
        res = json.load(open(f))
        panels = [n for n in ("loopmoe", "loopmoe_k1", "plainloop") if n in res]
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), sharey=True, squeeze=False,
                                 gridspec_kw={"width_ratios": [len(res[n]["visits"]) for n in panels]})
        titles = {"loopmoe": "looped MoE (K=3)", "loopmoe_k1": "looped MoE trained at K=1",
                  "plainloop": "plain loop, one residual stream"}
        labels = {"act_in": "logit lens (active block)", "mlp_in": "visit's own MoE-input mix",
                  "mix": "mix lens (final_input)"}
        for ax, name in zip(axes[0], panels):
            r = res[name]
            for k in ("act_in", "mlp_in", "mix"):
                if k in r:
                    lab = "logit lens (residual stream)" if name == "plainloop" else labels[k]
                    ax.plot(r[k], marker=".", label=lab)
            ax.set_xticks(range(len(r["visits"])), r["visits"], rotation=70, fontsize=7)
            ax.set(title=titles[name], ylim=(0, 1.02))
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("top-1 is the answer at '?'")
        fig.tight_layout()
        fig.savefig(f"{path}/lens.png", dpi=150)
    print(f"wrote figures to {path}/")


# ----------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="train", choices=["train", "sample", "sweep", "compare", "score", "lens", "plot"])
    p.add_argument("--only", default="", help="compare: comma-separated subset of the five models")
    p.add_argument("--tag", default="", help="score/lens/plot: checkpoint suffix (long = out/chain_loopmoe_long.pt)")
    for f in fields(Config):  # every Config field is a flag, e.g. --loops 5 --mix 0 --n_shared 1
        cast = (lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type
        p.add_argument(f"--{f.name}", type=cast, default=None)
    args = vars(p.parse_args())
    mode, only, tag = args.pop("mode"), [s for s in args.pop("only").split(",") if s], args.pop("tag")
    overrides = {k: v for k, v in args.items() if v is not None}
    c = Config(**overrides)
    if mode == "train":
        model = train(c)
        if c.task == "chain":
            for K, a in sweep(model, c).items():
                print(f"K={K} mean {sum(a.values()) / len(a):.3f} | " + " ".join(f"{v:.2f}" for v in a.values()))
    elif mode == "compare":
        compare(c, only)
    elif mode == "score":  # re-score saved checkpoints: out/<task>_<name>[_<tag>].pt -> out/compare_<task>[_<tag>].json
        suffix = f"_{tag}" if tag else ""
        results = {}
        for name, _ in compare_configs(c):
            if os.path.exists(path := f"out/{c.task}_{name}{suffix}.pt"):
                model, mc = load(path, c.device)
                results[name] = score(model, mc)
        json.dump(results, open(f"out/compare_{c.task}{suffix}.json", "w"), indent=1)
        print_table(results)
    elif mode == "sweep":
        model, mc = load(c.out, c.device)
        for K, a in sweep(model, mc, 8).items():
            print(f"K={K} mean {sum(a.values()) / len(a):.3f} | " + " ".join(f"{v:.2f}" for v in a.values()))
    elif mode == "lens":  # reads out/chain_<name>[_<tag>].pt from --mode compare, writes out/lens[_<tag>].json
        suffix, res = (f"_{tag}" if tag else ""), {}
        for name in ("loopmoe", "loopmoe_k1", "plainloop"):
            if os.path.exists(path := f"out/chain_{name}{suffix}.pt"):
                model, mc = load(path, c.device)
                res[name] = lens(model, mc, overrides.get("loops"))
                print(f"\n{name} ({path})")
                print_lens(res[name])
        os.makedirs("out", exist_ok=True)
        json.dump(res, open(f"out/lens{suffix}.json", "w"), indent=1)
    elif mode == "plot":
        plot(f"_{tag}" if tag else "")
    else:  # sample
        model, mc = load(c.out, c.device)
        dev = next(model.parameters()).device
        K = overrides.get("loops", mc.loops)
        if mc.task == "chain":
            rng = random.Random(c.seed)
            for _ in range(3):
                text, qs = chain_example(rng, mc.hops)
                off = mc.block_size - len(text)
                x = torch.tensor([[mc.vocab.index(ch) for ch in text.rjust(mc.block_size, ".")]], device=dev)
                guess = model(x, K)[0][0].argmax(-1).tolist()
                print(f"{text}   (K={K})")
                for i, digit, h in qs:
                    g = mc.vocab[guess[off + i]]
                    print(f"  {text[i - 1]}? -> {g}  truth {digit}  ({h}-hop{'' if g == digit else '  WRONG'})")
        else:
            torch.manual_seed(c.seed)
            idx = torch.tensor([[mc.vocab.index("\n")]], device=dev)
            for _ in range(500):
                logits, _ = model(idx[:, -mc.block_size:], K)
                idx = torch.cat([idx, torch.multinomial(F.softmax(logits[:, -1] / 0.8, -1), 1)], 1)
            print("".join(mc.vocab[i] for i in idx[0].tolist()))


if __name__ == "__main__":
    main()
