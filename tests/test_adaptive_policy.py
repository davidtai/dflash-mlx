from types import SimpleNamespace

from dflash_mlx.engine.spec_epoch import (
    _AdaptiveBlockPolicy,
    _resolve_adaptive_block_policy,
)


def test_external_adaptive_policy_factory_overrides_stock_policy() -> None:
    expected = SimpleNamespace(
        block_limit=lambda: 3,
        record=lambda **_kwargs: None,
        reductions=0,
        reduced_cycles=0,
        min_seen=1,
        metrics=lambda: {"kind": "external"},
    )
    calls = []
    target_model = SimpleNamespace(
        _dflash_adaptive_block_policy_factory=lambda **kwargs: calls.append(kwargs)
        or expected
    )
    runtime_config = SimpleNamespace(verify_mode="dflash")

    policy = _resolve_adaptive_block_policy(
        target_model=target_model,
        runtime_config=runtime_config,
        effective_block_tokens=8,
        verify_len_cap=8,
        prompt_len=16_384,
    )

    assert policy is expected
    assert calls == [
        {
            "full_block_tokens": 8,
            "verify_len_cap": 8,
            "prompt_len": 16_384,
        }
    ]


def test_adaptive_m4_full_accept_uses_draft_capacity() -> None:
    policy = _AdaptiveBlockPolicy(full_block_tokens=16)

    policy.record(block_len=4, acceptance_len=3, cycle_cost_ns=1_000_000)

    metrics = policy.metrics()
    assert metrics["full_accept_rate_by_block"]["4"] == 1.0
    assert metrics["acceptance_by_block"]["4"] == 1.0


def test_adaptive_probe_resume_can_pass_speed_ratio() -> None:
    policy = _AdaptiveBlockPolicy(
        full_block_tokens=16,
        early_probe_cycles=1,
        early_probe_min_cycles=2,
        early_probe_window=2,
        resume_speed_ratio=1.1,
    )
    policy._enter_reduced()

    policy.record(block_len=4, acceptance_len=3, cycle_cost_ns=10_000_000)
    policy.record(block_len=4, acceptance_len=3, cycle_cost_ns=10_000_000)
    assert policy.mode == "probe"

    policy.record(block_len=16, acceptance_len=8, cycle_cost_ns=20_000_000)
    assert policy.mode == "full"


def test_adaptive_normal_probe_rejects_slow_full_block() -> None:
    policy = _AdaptiveBlockPolicy(
        full_block_tokens=16,
        early_probe_min_cycles=0,
        resume_speed_ratio=1.1,
    )
    policy._enter_reduced()

    for _ in range(64):
        policy.record(block_len=4, acceptance_len=2, cycle_cost_ns=10_000_000)
    assert policy.mode == "probe"
    assert policy.probe_is_early is False

    for _ in range(8):
        policy.record(block_len=16, acceptance_len=8, cycle_cost_ns=40_000_000)
    assert policy.mode == "reduced"


def test_adaptive_early_probe_keeps_speed_gate() -> None:
    policy = _AdaptiveBlockPolicy(
        full_block_tokens=16,
        early_probe_cycles=1,
        early_probe_min_cycles=2,
        early_probe_window=2,
        resume_speed_ratio=1.1,
    )
    policy._enter_reduced()

    policy.record(block_len=4, acceptance_len=3, cycle_cost_ns=10_000_000)
    policy.record(block_len=4, acceptance_len=3, cycle_cost_ns=10_000_000)
    assert policy.mode == "probe"
    assert policy.probe_is_early is True

    policy.record(block_len=16, acceptance_len=8, cycle_cost_ns=40_000_000)

    assert policy.mode == "reduced"
    assert policy.early_probe_lockout is True


def test_adaptive_normal_probe_keeps_speed_gate() -> None:
    policy = _AdaptiveBlockPolicy(
        full_block_tokens=16,
        early_probe_cycles=1,
        early_probe_min_cycles=0,
        resume_speed_ratio=1.1,
    )
    policy._enter_reduced()

    for _ in range(64):
        policy.record(block_len=4, acceptance_len=2, cycle_cost_ns=10_000_000)

    assert policy.mode == "probe"
    assert policy.probe_is_early is False

    for _ in range(7):
        policy.record(block_len=16, acceptance_len=8, cycle_cost_ns=40_000_000)

    assert policy.mode == "probe"
    policy.record(block_len=16, acceptance_len=8, cycle_cost_ns=40_000_000)

    assert policy.mode == "reduced"


def test_adaptive_speed_gate_uses_current_reduced_episode() -> None:
    policy = _AdaptiveBlockPolicy(
        full_block_tokens=16,
        early_probe_cycles=1,
        early_probe_min_cycles=0,
        resume_speed_ratio=1.1,
    )
    policy._enter_reduced()

    for _ in range(64):
        policy.record(block_len=4, acceptance_len=2, cycle_cost_ns=40_000_000)
    for _ in range(8):
        policy.record(block_len=16, acceptance_len=8, cycle_cost_ns=20_000_000)
    assert policy.mode == "full"

    policy._enter_reduced()
    for _ in range(64):
        policy.record(block_len=4, acceptance_len=2, cycle_cost_ns=10_000_000)
    assert policy.mode == "probe"

    for _ in range(8):
        policy.record(block_len=16, acceptance_len=8, cycle_cost_ns=40_000_000)
    assert policy.mode == "reduced"
