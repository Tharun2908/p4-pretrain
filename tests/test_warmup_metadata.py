"""Regression test for windowed benchmark warmup metadata."""

from train import main


def test_windowed_warmup_metadata_cpu():
    result = main([
        "--model", "gpt-tiny",
        "--device", "cpu",
        "--strategy", "ddp",
        "--micro-bs", "1",
        "--steps", "4",
        "--bench-warmup", "2",
        "--log-every", "1",
        "--dry-run",
    ])

    assert result["warmup_steps_discarded"] == 2
    assert result["measured_steps"] == 2
    assert result["timing_mode"] == "windowed"
    assert result["tokens_per_sec"] > 0
