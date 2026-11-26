# nanoloop-moe

A small, readable looped mixture-of-experts transformer in the style of Inkling 2 Mini. One file of PyTorch
(`nanoloop_moe.py`): a prelude, a core that runs K times with shared weights, a coda, a top-k MoE in every layer,
**no single residual stream** (each sublayer reads its own learned mix of finished blocks), and a **gated message**
from one core pass to the next. You choose K at inference.

> An Inkling-style toy built from public sources. Not affiliated with, endorsed by or derived from the weights of
> Thinking Machines. "Inkling" is Thinking Machines' product name.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albertlieyingadrian/nanoloop-moe/blob/main/notebooks/quickstart.ipynb)

```
 ordinary transformer: one running sum             this model: a list of finished blocks

  x = embed                                        blocks = [embed]        h = the active block
  x = x + attn(x)                                  every sublayer reads mix(blocks + [h]):
  x = x + ffn(x)        (every layer adds          its own learned softmax over blocks
  ...                    to the same x)            at a block boundary h is frozen into blocks
  logits = head(x)                                 logits = head(final_input mix(blocks + [h]))

 the loop (3 core passes):

  embed -> [prelude] --+--> [core pass 1] --+       each pass restarts from the prelude's blocks;
                       |                    | bank  the previous pass reaches the next one only as
                       +--> [core pass 2] <-+--+    a message: x <- (1 - g_forget) * x + g_message * msg
                       |                       | bank
                       +--> [core pass 3] <----+
                                  |
                               [coda] -> final_input mix -> logits
```

## Quickstart

```bash
git clone https://github.com/albertlieyingadrian/nanoloop-moe && cd nanoloop-moe
pip install -r requirements.txt                 # torch, numpy, matplotlib (plots only)
python test_nanoloop_moe.py                     # 12 checks, about 2 s on CPU
python nanoloop_moe.py --mode compare           # chain task, 5 models (GPU: ~15 min each; see below for CPU)
python nanoloop_moe.py --mode sample --out out/chain_loopmoe.pt --loops 1   # same weights, 1 core pass
python nanoloop_moe.py --mode sample --out out/chain_loopmoe.pt --loops 3   # and 3
python nanoloop_moe.py --mode lens              # logit lens vs mix lens, per layer visit
python nanoloop_moe.py --mode plot              # assets/k_sweep.png and assets/lens.png
python nanoloop_moe.py --mode compare --task shakespeare --steps 3000
modal run modal_run.py                          # the same compare on Modal H100s, one container per model
```

Every field of `Config` is a flag: `--loops 5 --mix 0 --messages 0 --moe 0 --n_shared 1 --block_layers 6 ...`.
On a laptop CPU the default chain runs take hours (the MoE runs every expert on every token, which is fast on a GPU
and slow on a CPU); `--n_embd 64 --steps 2000` or the notebook's small settings make a quick look.

RESULTS_PLACEHOLDER

## What is verified and what is assumed

Sources: the Hugging Face `transformers` implementation (`models/inkling2_loop`, `models/inkling2`), the model's
`config.json`, the model card, the release post and the nnsight post on reading the model (links in Credits).

**Verified, and copied in shape (at toy size):**

| Inkling 2 Mini | here (default) |
|---|---|
| 24 layers in 4 blocks of 6: prelude 0-5, core 6-17, coda 18-23 | 8 layers in 4 blocks of 2 (`--block_layers 6` gives the real layout; a test checks it has 48 visits) |
| the core runs 3 times with shared weights: 48 layer visits | K = 3 by default, any K at inference |
| no single residual stream: sublayer outputs add up only inside the active block; at a block boundary the block is frozen into a list | same |
| every attention and MoE sublayer reads a softmax mix over RMS-normalized blocks + the active block, scored against a learned query (attention residuals); queries start at zero | same (`DepthMix`) |
| full-attention layers use 3 queries (separate q, k, v mixes); the head reads its own `final_input` mix | 3 queries in every layer; `final_input` the same |
| each core pass restarts from the prelude's blocks; the previous pass's core blocks become the message bank | same |
| passes 2+ read 4 messages per core layer (q, k, v, MoE input) through learned queries | same (`MessageInput`) |
| gate: x <- (1 - g_forget) * x + g_message * message, tanh gates from a rank-64 projection of [norm x, norm message]; closed at the start of training | same, rank 16; gate projections zero-initialized |
| MoE in every layer, 8 of 128 experts per token, sigmoid router, a router bias used only for choosing experts, weights renormalized and scaled by 8 | 4 of 16, same router; scale = top_k |
| no shared expert (`n_shared_experts: 0`) | 0 by default; `--n_shared 1` adds one |
| softcapped SwiGLU experts (gate cap 4, up cap 25), final logit softcap 30 | same |
| each (layer, pass) visit has its own KV cache | no cache here; full-sequence training gives the same attention pattern |

**Assumed or simplified (stand-ins, not Inkling's design):**

- **Load-balancing update.** The router bias exists in the released code (`e_score_correction_bias`) but the rule that
  trains it is not public. Here it is the auxiliary-loss-free rule from DeepSeek-V3: after every step, raise the bias
  of experts that got fewer tokens than average and lower it for the others, by a fixed rate.
- **Positions and attention.** Inkling mixes local (sliding-window) and global layers and uses a learned relative
  position bias, short convolutions after each sublayer, L2-clipped queries and keys, and a gated output projection.
  Here every layer is plain causal attention with rotary positions, and there are no convolutions. At these context
  lengths (96 characters) a 1024-token window would cover everything anyway.
- **Sizes.** Width 128, 4 heads, 16 experts of width 64, 2 layers per block, gate rank 16. Chosen so a run takes
  minutes on one GPU.
- **Training.** Inkling's data, schedule, pruning (it started with 384 experts and kept 128) and post-training are not
  reproduced. Here: AdamW at 5e-4, gradient clipping at 1, a curriculum on the chain task.
- **Shared expert.** Optional, off by default, because Inkling 2 Mini's config has none.

## Files

- `nanoloop_moe.py`: model, baselines, data, training, K sweep, lens, plots, CLI.
- `test_nanoloop_moe.py`: 12 CPU checks (shapes, the 48-visit layout, passes restart from the prelude, the gate starts
  closed, mixing at zero queries is an average and with `--mix 0` is the residual stream, routing and balancing, the
  stacked MoE equals routing token by token, matched active parameters, lens and sweep, chain answers, save and load).
- `modal_run.py`: the compare on Modal, one H100 container per model.
- `notebooks/quickstart.ipynb`: tests, a short compare, K at inference, the lens.
- `results/`: the JSON behind every number in this README.

## Related

- nanoloop-transformer, the plain looped transformer this grew from (one residual stream, input injection):
  https://github.com/albertlieyingadrian/nanoloop-transformer
- Build a Looped Transformer, the stage-by-stage course: https://github.com/albertlieyingadrian/build-a-looped-transformer
- Build a Looped MoE, the stage-by-stage course that ends with this file: https://github.com/albertlieyingadrian/build-a-looped-moe

## Credits and further reading

- Inkling 2 Mini: the release post, https://thinkingmachines.ai/news/inkling-2-mini/, the model card,
  https://huggingface.co/thinkingmachines/Inkling-2-Mini, and its implementation in Hugging Face `transformers`
  (`models/inkling2_loop`), https://github.com/huggingface/transformers.
- Opening the box of Inkling 2 Mini using nnsight and NDIF, https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/:
  the block list, the message bank and the gates, with module names.
- The Looped Lens Challenge (NDIF), https://ndif.us/looped-lens-challenge/: the question the lens mode is a toy version of.
- Attention Residuals (2026), https://arxiv.org/abs/2603.15031: the depth-attention mixing Inkling follows.
- DenseFormer (2024), https://arxiv.org/abs/2402.02622, and Hyper-Connections (2024), https://arxiv.org/abs/2409.19606:
  earlier ways to let layers read a learned mix of earlier outputs.
- Universal Transformers (2018), https://arxiv.org/abs/1807.03819: shared weights across depth.
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (2025), https://arxiv.org/abs/2502.05171:
  prelude / recurrent core / coda.
- Mixture of experts: Outrageously Large Neural Networks (2017), https://arxiv.org/abs/1701.06538; GShard (2020),
  https://arxiv.org/abs/2006.16668; Switch Transformers (2021), https://arxiv.org/abs/2101.03961; DeepSeekMoE (2024),
  https://arxiv.org/abs/2401.06066; DeepSeek-V3 (2024), https://arxiv.org/abs/2412.19437 (sigmoid routing and
  auxiliary-loss-free balancing).
- The shape of this repo: nanoGPT, https://github.com/karpathy/nanoGPT. No code is copied from it. The chain task
  and the plain-loop baseline come from nanoloop-transformer (same licence).
- Tiny Shakespeare from the char-rnn repo, https://github.com/karpathy/char-rnn, downloaded on first use.

## License

MIT
