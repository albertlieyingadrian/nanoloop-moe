"""Fast CPU checks. Run with `python test_nanoloop_moe.py` or `pytest`."""
import random
import tempfile

import torch

import nanoloop_moe as nm


def tiny(**kw):
    base = dict(vocab=nm.CHAIN_VOCAB, n_embd=32, n_head=2, block_layers=2, n_experts=4, top_k=2,
                d_expert=16, gate_rank=4, hops=3, block_size=40, batch_size=4)
    return nm.Config(**{**base, **kw})


def test_shapes_all_archs():
    for arch, moe in (("loopmoe", True), ("plainloop", False), ("standard", False)):
        c = tiny(arch=arch, moe=moe)
        x, y, hops = nm.chain_batch(random.Random(0), c)
        logits, loss = nm.build(c)(x, targets=y)
        assert logits.shape == (4, 40, len(nm.CHAIN_VOCAB)) and torch.isfinite(loss)
        assert ((y >= 0) == (hops > 0)).all()  # only answers are scored


def test_loops_add_compute_not_weights():
    n = lambda c: sum(p.numel() for p in nm.build(c).parameters())  # noqa: E731
    assert n(tiny(loops=1)) == n(tiny(loops=6))


def test_inkling_layout_has_48_visits():
    # 24 layers in blocks of 6: prelude 0-5, core 6-17 run 3 times, coda 18-23.
    c = tiny(block_layers=6, prelude_blocks=1, core_blocks=2, coda_blocks=1, loops=3, n_embd=16, n_head=2)
    m = nm.LoopMoE(c)
    v = m.visits(3)
    assert len(m.layers) == 24 and len(v) == 48
    assert v[6] == (6, 0) and v[18] == (6, 1) and v[41] == (17, 2) and v[42] == (18, 0)
    rec = []
    m(torch.zeros(1, 8, dtype=torch.long), rec=rec)
    assert len(rec) == 48 + 1  # every visit plus the final read
    # the blocks the head mixes: embeddings, prelude, core block A, core block B (+ the active coda block)
    assert rec[-1]["blocks"].shape[-2] == 4


def test_each_pass_restarts_from_the_prelude():
    # Without messages nothing carries over between passes, so K does not change the output.
    torch.manual_seed(0)
    m = nm.LoopMoE(tiny(messages=False)).eval()
    x = torch.randint(0, len(nm.CHAIN_VOCAB), (2, 40))
    assert torch.allclose(m(x, 1)[0], m(x, 4)[0], atol=1e-5)
    # With messages, later passes read the previous one (once the gates are not all zero).
    m = nm.LoopMoE(tiny()).eval()
    for layer in m.layers:
        if layer.gates is not None:
            for g in layer.gates:
                torch.nn.init.normal_(g.message.weight, std=0.5)
    assert not torch.allclose(m(x, 1)[0], m(x, 3)[0], atol=1e-4)


def test_message_gate_starts_closed():
    g = nm.MessageGate(tiny())
    x, msg = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    assert torch.allclose(g(x, msg), x)


def test_depth_mix():
    blocks, active = torch.randn(2, 3, 4, 32), torch.randn(2, 3, 32)
    _, w = nm.DepthMix(tiny(), n_query=3).weights(blocks, active)
    assert w.shape == (2, 3, 5, 3) and torch.allclose(w, torch.full_like(w, 0.2))  # zero query = average
    s = nm.DepthMix(tiny(mix=False))(blocks, active)[0]
    assert torch.allclose(s, blocks.sum(-2) + active)  # no mixing = the ordinary residual stream


def test_router_and_balance():
    torch.manual_seed(0)
    moe = nm.MoE(tiny(n_experts=8, top_k=2))
    w, idx = moe.route(torch.randn(50, 32))
    assert ((w > 0).sum(-1) == 2).all() and torch.allclose(w.sum(-1), torch.full((50,), 2.0))
    moe.load = torch.tensor([10.0, 0, 0, 0, 0, 0, 0, 30.0])
    moe.balance(0.1)
    assert moe.bias[7] < 0 and moe.bias[1] > 0  # over-used experts go down, idle ones go up


def test_moe_matches_per_expert_loop():
    # Running every expert and zero-weighting the unchosen ones equals routing token by token.
    torch.manual_seed(0)
    moe = nm.MoE(tiny(n_experts=4, top_k=2, n_shared=1)).eval()
    x = torch.randn(6, 32)
    w, idx = moe.route(x)
    ref = moe.shared(x).clone()
    for n in range(6):
        for e in idx[n].tolist():
            g = x[n] @ moe.w_gate[e]
            h = nm.softcap(g, 4.0) * torch.sigmoid(g) * nm.softcap(x[n] @ moe.w_up[e], 25.0)
            ref[n] += w[n, e] * (h @ moe.w_down[e])
    assert torch.allclose(moe(x), ref, atol=1e-5)


def test_active_params_match_dense_baseline():
    moe_total, moe_active = nm.count_params(nm.build(tiny()))
    dense_total, dense_active = nm.count_params(nm.build(tiny(arch="plainloop", moe=False)))
    assert moe_total > moe_active and dense_total == dense_active
    # what is left is the mixing queries, message gates and routers: small next to the experts
    assert abs(moe_active - dense_active) / dense_active < 0.25


def test_lens_and_sweep_smoke():
    c = tiny()
    out = nm.lens(nm.build(c), c, n=8)
    assert len(out["visits"]) == len(out["act_in"]) == len(out["mix"]) == 2 + 3 * 4 + 2 + 1
    pc = tiny(arch="plainloop", moe=False)
    assert len(nm.lens(nm.build(pc), pc, n=8)["act_in"]) == 3 * 4 + 2 + 1
    accs = nm.sweep(nm.build(c), c, max_loops=2)
    assert set(accs) == {1, 2} and all(0 <= a <= 1 for a in accs[1].values())


def test_chain_answer_is_reachable():
    text, qs = nm.chain_example(random.Random(3), max_hops=3)
    table = dict(s.split("=") for s in text.split(";")[:-1])
    for i, digit, hops in qs:
        assert text[i] == "?"
        v = text[i - 1]
        for _ in range(hops):
            v = table[v]
        assert v == digit


def test_train_two_steps_and_reload():
    out = f"{tempfile.mkdtemp()}/m.pt"
    nm.train(tiny(steps=2, device="cpu", out=out), log=lambda *_: None)
    model, c = nm.load(out, "cpu")
    assert c.loops == 3 and isinstance(model, nm.LoopMoE)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
