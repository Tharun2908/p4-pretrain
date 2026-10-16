"""
p4/tests/test_model.py — does model.py compute what we think it computes?

    pytest -q                 # CPU, ~seconds; runs on the laptop
    pytest -q -k gpu          # GPU-only backend tests; run in the pod on 1×B200

Each test guards against one specific way a GPT can be silently wrong — i.e. train, produce
a loss curve, and still be broken. A model that crashes is easy; these catch the ones that don't.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from model import GPT
from model_configs import MODELS, GPTConfig

TINY = GPTConfig("tiny", n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=128)


def meta_model(cfg: GPTConfig) -> GPT:
    """Build on the 'meta' device: shapes only, no memory, no init compute — so counting
    the 1.3B model's parameters costs nothing on a laptop."""
    with torch.device("meta"):
        return GPT(cfg)


# ---- 1. parameter counts ------------------------------------------------------------------

@pytest.mark.parametrize("cfg, published", [
    (GPTConfig("gpt2", 12, 12, 768, vocab_size=50257), 124_439_808),
    (GPTConfig("gpt2-medium", 24, 16, 1024, vocab_size=50257), 354_823_168),
])
def test_matches_published_gpt2_param_counts(cfg, published):
    """External anchor: GPT-2's published counts. Catches a missing bias, an untied head
    (+38.6M), a wrong MLP ratio — anything structural — against a number we didn't derive."""
    assert meta_model(cfg).num_params(non_embedding=False) == published


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("bias", [True, False])
def test_model_matches_config_arithmetic(name, bias):
    """model_configs.py does the arithmetic MFU depends on; model.py builds the real thing.
    If they disagree, every MFU number in the project is wrong."""
    cfg = GPTConfig(**{**MODELS[name].__dict__, "bias": bias})
    m = meta_model(cfg)
    assert m.num_params(non_embedding=False) == cfg.n_params
    assert m.num_params(non_embedding=True) == cfg.n_params_non_embedding


def test_vocab_padded_to_multiple_of_64():
    for cfg in MODELS.values():
        assert cfg.vocab_size % 64 == 0 and cfg.vocab_size >= 50257


# ---- 2. weight tying ----------------------------------------------------------------------

def test_weight_tying_is_one_tensor_with_two_gradient_paths():
    torch.manual_seed(0)
    m = GPT(TINY)
    wte, head = m.transformer.wte.weight, m.lm_head.weight
    assert head is wte, "tie broken: two tensors, not one"

    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    _, loss = m(idx, idx)
    loss.backward()
    # Rows for tokens NOT in the input get no embedding gradient — only lm_head gradient
    # (every row of the output projection gets softmax gradient). A nonzero grad on an
    # unseen token's row therefore proves the output path reaches the shared tensor.
    unseen = torch.ones(TINY.vocab_size, dtype=torch.bool)
    unseen[idx.unique()] = False
    assert wte.grad[unseen].abs().sum() > 0


# ---- 3. causality -------------------------------------------------------------------------

def test_future_tokens_cannot_influence_past_logits():
    """The causal mask is the entire difference between a language model and a cheat. If it
    leaks, training loss collapses to ~0 because the model reads the answer — and looks GREAT."""
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 24))
    t = 12
    idx2 = idx.clone()
    idx2[0, t] = (idx[0, t] + 1) % TINY.vocab_size
    with torch.no_grad():
        a, _ = m(idx)
        b, _ = m(idx2)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6), "future token leaked into the past"
    assert not torch.allclose(a[:, t:], b[:, t:]), "changed token had no effect at all"


# ---- 4. initialization --------------------------------------------------------------------

def test_initial_loss_is_near_uniform():
    """At init the model should know nothing: loss ≈ ln(V). Much higher = logits too large at
    init (bad std / broken LN) → early instability. Use the REAL vocab, small width.

    Targets MUST be next-token (y = idx[:, 1:]), not the inputs themselves. With tied
    embeddings the logit for the CURRENT token is ≈ |wte[t]|² / std(residual) ≈ 0.02·C/√2,
    a built-in copy bias that grows linearly with width (~0.9 at C=64, ~11 at C=768).
    Scoring inputs-as-targets measures that copy bias, not initialization quality.
    (Original version of this test made exactly that mistake — caught in external review.)

    For next-token targets: expected ≈ ln(V) + σ²/2, logit std σ ≈ 0.02·sqrt(C) = 0.16 here."""
    torch.manual_seed(0)
    cfg = GPTConfig("init", n_layer=2, n_head=4, n_embd=64, block_size=64)   # V = 50304
    m = GPT(cfg)
    idx = torch.randint(0, 50257, (4, 64))
    x, y = idx[:, :-1], idx[:, 1:]            # non-contiguous on purpose: exercises reshape
    with torch.no_grad():
        _, loss = m(x, y)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.1, loss.item()


def test_tied_embeddings_carry_a_copy_bias_at_init():
    """Documents the structural effect above rather than just avoiding it: scoring the
    current token as the target gives a loss well BELOW uniform at init."""
    torch.manual_seed(0)
    cfg = GPTConfig("init", n_layer=2, n_head=4, n_embd=64, block_size=64)
    m = GPT(cfg)
    idx = torch.randint(0, 50257, (4, 64))
    with torch.no_grad():
        _, copy_loss = m(idx, idx)
    assert copy_loss.item() < math.log(cfg.vocab_size) - 0.3, copy_loss.item()


def test_residual_projections_get_depth_scaled_init():
    torch.manual_seed(0)
    cfg = GPTConfig("deep", n_layer=8, n_head=4, n_embd=256, block_size=32, vocab_size=128)
    m = GPT(cfg)
    want = 0.02 / math.sqrt(2 * cfg.n_layer)
    for blk in m.transformer.h:
        for proj in (blk.attn.c_proj, blk.mlp.c_proj):
            assert abs(proj.weight.std().item() - want) / want < 0.05
        assert abs(blk.mlp.c_fc.weight.std().item() - 0.02) / 0.02 < 0.05   # unscaled


# ---- 5. gradient flow end to end ----------------------------------------------------------

def test_can_overfit_a_single_batch():
    """If the model can't memorize 4 sequences of 32 tokens, gradients don't flow somewhere —
    a detached tensor, a frozen param, a broken residual. The cheapest end-to-end test there is."""
    torch.manual_seed(0)
    m = GPT(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(0, TINY.vocab_size, (4, 32))
    x, y = idx[:, :-1], idx[:, 1:]
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()


def test_ignore_index_masks_positions():
    torch.manual_seed(0)
    m = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (1, 16))
    y = idx.clone()
    y[:, 8:] = -1
    logits, loss = m(idx, y)
    ref = F.cross_entropy(logits[0, :8], idx[0, :8])
    assert torch.allclose(loss, ref)


def test_rejects_sequence_longer_than_block_size():
    with pytest.raises(ValueError):
        GPT(TINY)(torch.zeros(1, TINY.block_size + 1, dtype=torch.long))


# ---- 6. attention against an independent reference ----------------------------------------

def reference_attention(attn_module, x):
    """Attention written out longhand: explicit q·kᵀ, scale, -inf mask, softmax, ·v.
    Shares only the WEIGHTS with model.py — none of its reshaping or the SDPA call."""
    B, T, C = x.shape
    nh, hd = attn_module.n_head, attn_module.head_dim
    q, k, v = attn_module.c_attn(x).split(C, dim=2)
    q, k, v = (t.reshape(B, T, nh, hd).permute(0, 2, 1, 3) for t in (q, k, v))
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
    y = (att @ v).permute(0, 2, 1, 3).reshape(B, T, C)
    return attn_module.c_proj(y)


def test_sdpa_attention_matches_longhand_reference():
    torch.manual_seed(0)
    attn = GPT(TINY).transformer.h[0].attn
    x = torch.randn(2, 20, TINY.n_embd)
    with torch.no_grad():
        assert torch.allclose(attn(x), reference_attention(attn, x), atol=1e-5)


# ---- 7. GPU backends (pod only) -----------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("backend", ["FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION"])
def test_gpu_backend_error_within_bf16_floor(backend):
    """
    Is each fused backend as accurate as bf16 itself allows — forward AND backward — at the
    1.3B model's attention shape?

    Three runs, one variable changed at a time:
        fp32 MATH   ground truth
        bf16 MATH   measures the PRECISION FLOOR: what bf16 alone costs vs fp32
        bf16 <be>   the backend under test
    Assert: error(be vs fp32) ≤ 2 × floor (+ tiny abs slack). Comparing be directly against
    bf16 MATH would NOT isolate backend error — two bf16 paths round in different places and
    disagree by about the floor even when both are correct. The tolerance is calibrated
    against a measurement instead of guessed.  Run with  pytest -q -s -k gpu  to see numbers.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.manual_seed(0)
    cfg = GPTConfig("probe", n_layer=1, n_head=16, n_embd=2048, block_size=1024)
    attn = GPT(cfg).transformer.h[0].attn.cuda()
    x = torch.randn(2, 1024, cfg.n_embd, device="cuda")

    def run(be_name: str, bf16: bool):
        xi = x.clone().requires_grad_()
        with sdpa_kernel(getattr(SDPBackend, be_name)), \
             torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            y = attn(xi)
            y.float().square().mean().backward()
        return y.detach().float(), xi.grad.float()

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    y_ref, g_ref = run("MATH", bf16=False)
    y_mb, g_mb = run("MATH", bf16=True)
    y_be, g_be = run(backend, bf16=True)

    floor_f, floor_b = rel(y_mb, y_ref), rel(g_mb, g_ref)
    err_f, err_b = rel(y_be, y_ref), rel(g_be, g_ref)
    print(f"\n  {backend:<20} fwd err {err_f:.2e} (floor {floor_f:.2e}) | "
          f"bwd err {err_b:.2e} (floor {floor_b:.2e})")

    assert floor_f < 5e-2 and floor_b < 1e-1, "bf16 floor itself is implausibly large"
    assert err_f <= 2 * floor_f + 1e-3, f"{backend} forward worse than bf16 warrants"
    assert err_b <= 2 * floor_b + 1e-3, f"{backend} backward worse than bf16 warrants"
