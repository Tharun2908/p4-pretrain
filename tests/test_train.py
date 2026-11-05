"""
p4/tests/test_train.py — train.py on CPU, before it touches a GPU.

Covers what can break silently or waste a pod session: the data shift, the guards, the
schedule, the optimizer groups, and an end-to-end smoke run. CUDA-only paths (autocast,
sdpa_kernel, compile, profiler) are exercised in the pod's first dry run.
"""

import math

import numpy as np
import pytest
import torch

import train
from model import GPT
from model_configs import DEBUG_MODELS

SMOKE = ["--model", "gpt-tiny", "--device", "cpu", "--micro-bs", "4",
         "--steps", "40", "--bench-warmup", "20", "--log-every", "10",
         "--lr", "3e-3", "--lr-warmup", "5", "--dry-run"]


def test_cpu_smoke_run_learns_and_measures():
    out = train.main(SMOKE)
    assert out["timing_mode"] == "windowed"
    assert out["measured_steps"] == 20            # steps 21..40, i.e. windows ending 30 and 40
    assert out["tokens_per_sec"] > 0
    assert out["mfu"] is None                       # CPU: not applicable, not a fake percentage
    # synthetic pool = 8 fixed batches, so even 40 steps must memorize SOMETHING
    assert out["final_loss"] < out["first_loss"], (out["first_loss"], out["final_loss"])


def test_measured_run_without_prediction_is_refused():
    argv = [a for a in SMOKE if a != "--dry-run"] + ["--run-id", "X-001"]
    with pytest.raises(SystemExit):
        train.main(argv)


def test_steps_must_cover_warmup_plus_one_window():
    argv = SMOKE.copy()
    argv[argv.index("--steps") + 1] = "25"
    with pytest.raises(SystemExit):
        train.main(argv)


def test_memmap_targets_are_inputs_shifted_by_exactly_one(tmp_path):
    """The silent killer: an off-by-one here trains a model that predicts the CURRENT
    token — which, with tied embeddings, it's already biased to do. Loss looks great."""
    path = tmp_path / "tok.bin"
    np.arange(5000, dtype=np.uint16).tofile(path)
    b = train.MemmapBatches(str(path), micro_bs=6, block=64, device="cpu", seed=0)
    for _ in range(5):
        x, y = b.next()
        assert x.shape == y.shape == (6, 64) and x.dtype == torch.int64
        assert torch.equal(y, x + 1)


def test_memmap_rejects_file_shorter_than_a_window(tmp_path):
    path = tmp_path / "short.bin"
    np.arange(10, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        train.MemmapBatches(str(path), micro_bs=1, block=64, device="cpu")


def test_synthetic_batches_never_emit_vocab_padding():
    b = train.SyntheticBatches(micro_bs=4, block=64, device="cpu")
    for _ in range(16):
        x, y = b.next()
        assert x.max() < train.REAL_VOCAB and y.max() < train.REAL_VOCAB
        assert x.is_contiguous() and y.is_contiguous()


def test_lr_schedule_shape():
    lrs = [train.lr_at(s, 1e-3, warmup=10, total=100) for s in range(101)]
    assert all(a < b for a, b in zip(lrs[:10], lrs[1:10]))       # warmup rises
    assert math.isclose(max(lrs), 1e-3)
    assert math.isclose(lrs[-1], 1e-4)                            # cosine floor = 0.1·max
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))       # never rises after warmup


def test_weight_decay_only_on_matrices_and_tied_tensor_once():
    m = GPT(DEBUG_MODELS["gpt-tiny"])
    opt = train.build_optimizer(m, 1e-3, 0.1, "cpu")
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears twice (tied weight double-counted?)"
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == m.num_params(False)


# ---- backend enforcement: pure logic, so it's testable on CPU ---------------------------

CUDNN_PROFILE = ["aten::_scaled_dot_product_cudnn_attention",
                 "aten::_scaled_dot_product_cudnn_attention_backward",
                 "cudnn_generated_fort_native_sdpa_sm100_flash_fprop"]   # illustrative: cuDNN kernels can say "flash"
MATH_PROFILE = ["aten::_scaled_dot_product_attention_math", "aten::_softmax", "aten::bmm"]


def test_verify_backend_confirms_the_requested_backend():
    train.verify_backend("cudnn", CUDNN_PROFILE)
    train.verify_backend("auto", CUDNN_PROFILE)
    train.verify_backend("math", MATH_PROFILE)


def test_verify_backend_is_not_fooled_by_kernel_names():
    """cuDNN's own kernels can contain the word 'flash'. A naive substring check on 'flash'
    would 'confirm' flash on a cuDNN run — the check keys on aten op names instead."""
    with pytest.raises(RuntimeError):
        train.verify_backend("flash", CUDNN_PROFILE)


def test_verify_backend_rejects_fused_attention_when_math_requested():
    with pytest.raises(RuntimeError):
        train.verify_backend("math", CUDNN_PROFILE)


def test_verify_backend_rejects_empty_profile_for_explicit_backend():
    with pytest.raises(RuntimeError):
        train.verify_backend("cudnn", [])


def test_m6_precision_defaults_to_original_bf16():
    assert train.parse_args([]).precision == "bf16"


@pytest.mark.parametrize("precision", ["te-bf16", "fp8", "nvfp4"])
def test_m6_te_modes_reject_cpu(precision):
    with pytest.raises(SystemExit, match="require CUDA"):
        train.main(SMOKE + ["--precision", precision])
