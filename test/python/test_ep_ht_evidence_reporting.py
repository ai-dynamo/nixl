# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EP_TESTS_DIR = REPO_ROOT / "examples" / "device" / "ep" / "tests"
sys.path.insert(0, str(EP_TESTS_DIR))

from ht_target_selection import (  # noqa: E402
    CorrectnessPreflightError,
    EIGHT_GPU_SINGLE_NODE_TARGET,
    FOUR_GPU_SINGLE_NODE_TARGET,
    MULTI_NODE_TARGET,
    TargetSelectionError,
    classify_correctness_failure,
    format_correctness_evidence,
    validate_final_review_evidence,
)


def _args(**overrides: object) -> Namespace:
    values = {
        "correctness_target": FOUR_GPU_SINGLE_NODE_TARGET,
        "correctness_target_label": "four-GPU single-node HT correctness",
        "ci_correctness_only": True,
        "num_processes": 4,
        "num_tokens": 4096,
        "hidden": 7168,
        "num_topk": 8,
        "num_experts": 256,
    }
    values.update(overrides)
    return Namespace(**values)


def _fields(record: str) -> dict[str, str]:
    assert record.startswith("[evidence] ")
    return dict(
        field.split("=", 1) for field in record.removeprefix("[evidence] ").split()
    )


def test_four_gpu_success_evidence_names_result_topology_workload_and_scope() -> None:
    record = format_correctness_evidence(_args(), result="pass")

    fields = _fields(record)
    assert fields["schema"] == "ep_ht_correctness_evidence_v1"
    assert fields["evidence_id"] == "ep_ht_four_gpu_single_node"
    assert fields["target"] == FOUR_GPU_SINGLE_NODE_TARGET
    assert fields["target_label"] == "four_gpu_single_node_ht_correctness"
    assert fields["topology"] == "single_node"
    assert fields["world_size"] == "4"
    assert fields["num_nvl_ranks"] == "4"
    assert fields["num_rdma_ranks"] == "1"
    assert fields["workload"] == "tokens:4096,hidden:7168,top_k:8,experts:256"
    assert fields["workload_tokens"] == "4096"
    assert fields["hidden_size"] == "7168"
    assert fields["top_k"] == "8"
    assert fields["expert_count"] == "256"
    assert fields["correctness_only"] == "true"
    assert (
        fields["scope"]
        == "single_node_correctness_only_no_multi_node_rdma_no_performance"
    )
    assert fields["result"] == "pass"
    assert fields["status"] == "pass"
    assert fields["failure_category"] == "none"


def test_four_gpu_success_evidence_record_remains_unchanged() -> None:
    record = format_correctness_evidence(_args(), result="pass")

    assert record == (
        "[evidence] schema=ep_ht_correctness_evidence_v1 "
        "evidence_id=ep_ht_four_gpu_single_node "
        "target=four_gpu_single_node "
        "target_label=four_gpu_single_node_ht_correctness "
        "topology=single_node "
        "world_size=4 "
        "num_nvl_ranks=4 "
        "num_rdma_ranks=1 "
        "workload=tokens:4096,hidden:7168,top_k:8,experts:256 "
        "workload_tokens=4096 "
        "hidden_size=7168 "
        "top_k=8 "
        "expert_count=256 "
        "correctness_only=true "
        "scope=single_node_correctness_only_no_multi_node_rdma_no_performance "
        "result=pass "
        "status=pass "
        "failure_category=none"
    )


def test_four_gpu_failure_evidence_is_sanitized_and_categorized() -> None:
    sensitive_error = RuntimeError(
        "CUDA IPC failed at C:\\secret\\repo\\test_ht.py:123 "
        "tcp://worker.example.com:9999 ptr=0x7ffdeadbeef"
    )

    record = format_correctness_evidence(
        _args(),
        result="fail",
        failure_category=classify_correctness_failure(sensitive_error),
    )

    fields = _fields(record)
    assert fields["result"] == "fail"
    assert fields["status"] == "fail"
    assert fields["failure_category"] == "peer_wiring_failed"
    assert "secret" not in record
    assert "worker.example.com" not in record
    assert "9999" not in record
    assert "0x7ffdeadbeef" not in record


def test_four_gpu_preflight_failure_evidence_is_labeled_and_sanitized() -> None:
    sensitive_error = CorrectnessPreflightError(
        "runtime_not_ready",
        "runtime_not_ready: local TCPStore bind failed at "
        "C:\\secret\\repo\\test_ht.py:123 for tcp://worker.example.com:9999 "
        "ptr=0x7ffdeadbeef",
    )

    record = format_correctness_evidence(
        _args(),
        result="fail",
        failure_category=classify_correctness_failure(sensitive_error),
    )

    fields = _fields(record)
    assert fields["schema"] == "ep_ht_correctness_evidence_v1"
    assert fields["evidence_id"] == "ep_ht_four_gpu_single_node"
    assert fields["target"] == FOUR_GPU_SINGLE_NODE_TARGET
    assert fields["target_label"] == "four_gpu_single_node_ht_correctness"
    assert fields["topology"] == "single_node"
    assert fields["world_size"] == "4"
    assert fields["num_nvl_ranks"] == "4"
    assert fields["num_rdma_ranks"] == "1"
    assert fields["workload"] == "tokens:4096,hidden:7168,top_k:8,experts:256"
    assert fields["correctness_only"] == "true"
    assert (
        fields["scope"]
        == "single_node_correctness_only_no_multi_node_rdma_no_performance"
    )
    assert fields["result"] == "fail"
    assert fields["status"] == "fail"
    assert fields["failure_category"] == "runtime_not_ready"
    assert "secret" not in record
    assert "worker.example.com" not in record
    assert "9999" not in record
    assert "0x7ffdeadbeef" not in record


def test_failure_category_is_limited_to_reviewable_categories() -> None:
    record = format_correctness_evidence(
        _args(),
        result="fail",
        failure_category=(
            "runtime failed at C:\\secret\\repo\\test_ht.py:123 "
            "tcp://worker.example.com:9999 ptr=0x7ffdeadbeef"
        ),
    )

    fields = _fields(record)
    assert fields["failure_category"] == "correctness_failed"
    assert "secret" not in record
    assert "worker.example.com" not in record
    assert "9999" not in record
    assert "0x7ffdeadbeef" not in record


def test_target_selection_failures_are_classified_for_evidence() -> None:
    assert classify_correctness_failure(TargetSelectionError("bad target")) == (
        "unsupported_target"
    )


def test_eight_gpu_compatibility_evidence_is_distinct_from_four_gpu_target() -> None:
    record = format_correctness_evidence(
        _args(
            correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET,
            correctness_target_label="eight-GPU single-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=1,
        ),
        result="pass",
    )

    fields = _fields(record)
    assert fields["schema"] == "ep_ht_correctness_evidence_v1"
    assert fields["evidence_id"] == "ep_ht_eight_gpu_single_node_compatibility"
    assert fields["target"] == EIGHT_GPU_SINGLE_NODE_TARGET
    assert fields["target_label"] == "eight_gpu_single_node_ht_compatibility"
    assert fields["topology"] == "single_node"
    assert fields["world_size"] == "8"
    assert fields["num_nvl_ranks"] == "8"
    assert fields["num_rdma_ranks"] == "1"
    assert fields["correctness_only"] == "false"
    assert fields["scope"] == "established_eight_gpu_single_node_ht_compatibility"
    assert fields["result"] == "pass"
    assert fields["failure_category"] == "none"


def test_multi_node_compatibility_evidence_preserves_failure_semantics() -> None:
    record = format_correctness_evidence(
        _args(
            correctness_target=MULTI_NODE_TARGET,
            correctness_target_label="multi-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=2,
        ),
        result="fail",
        failure_category="runtime_not_ready",
    )

    fields = _fields(record)
    assert fields["evidence_id"] == "ep_ht_multi_node_compatibility"
    assert fields["target"] == MULTI_NODE_TARGET
    assert fields["target_label"] == "multi_node_ht_compatibility"
    assert fields["topology"] == "multi_node"
    assert fields["world_size"] == "16"
    assert fields["num_nvl_ranks"] == "8"
    assert fields["num_rdma_ranks"] == "2"
    assert fields["correctness_only"] == "false"
    assert fields["scope"] == "established_multi_node_ht_compatibility"
    assert fields["result"] == "fail"
    assert fields["status"] == "fail"
    assert fields["failure_category"] == "runtime_not_ready"


def test_ht_runner_emits_four_gpu_evidence_on_success_and_failure() -> None:
    ht_test = (EP_TESTS_DIR / "test_ht.py").read_text(encoding="utf-8")

    assert "format_correctness_evidence" in ht_test
    assert 'result="pass"' in ht_test
    assert 'result="fail"' in ht_test
    assert "classify_correctness_failure" in ht_test
    assert "FOUR_GPU_SINGLE_NODE_TARGET" in ht_test
    assert "correctness_target.name in EVIDENCE_CORRECTNESS_TARGETS" in ht_test


def test_ht_runner_emits_named_target_selection_failure_evidence() -> None:
    ht_test = (EP_TESTS_DIR / "test_ht.py").read_text(encoding="utf-8")
    handler_start = ht_test.index("except TargetSelectionError as exc:")
    parser_error = "        parser.error(str(exc))"

    target_selection_handler = ht_test[
        handler_start : ht_test.index(parser_error, handler_start) + len(parser_error)
    ]

    assert "SUPPORTED_CORRECTNESS_TARGETS" in target_selection_handler
    assert "format_correctness_evidence(" in target_selection_handler
    assert 'result="fail"' in target_selection_handler
    assert "classify_correctness_failure(exc)" in target_selection_handler
    assert "parser.error(str(exc))" in target_selection_handler


def test_ht_runner_emits_named_preflight_failure_evidence_through_exception_path() -> None:
    ht_test = (EP_TESTS_DIR / "test_ht.py").read_text(encoding="utf-8")

    preflight_index = ht_test.index("preflight_four_gpu_single_node_target(")
    tcpstore_index = ht_test.index('print("Starting TCPStore and rank server locally"')
    spawn_index = ht_test.index("torch.multiprocessing.spawn(")
    handler_start = ht_test.index("    except Exception as exc:", preflight_index)
    success_record_index = ht_test.index(
        "        print(format_correctness_evidence(args, result=\"pass\"), flush=True)",
        handler_start,
    )
    handler = ht_test[handler_start:success_record_index]

    assert preflight_index < tcpstore_index
    assert preflight_index < spawn_index
    assert "if correctness_target.name in EVIDENCE_CORRECTNESS_TARGETS:" in handler
    assert "format_correctness_evidence(" in handler
    assert 'result="fail"' in handler
    assert "failure_category=classify_correctness_failure(exc)" in handler
    assert handler.index("format_correctness_evidence(") < handler.index("sys.exit(1)")


def test_ht_runner_emits_named_compatibility_evidence_without_reduced_ci_mode() -> None:
    ht_test = (EP_TESTS_DIR / "test_ht.py").read_text(encoding="utf-8")

    assert "EIGHT_GPU_SINGLE_NODE_TARGET" in ht_test
    assert "MULTI_NODE_TARGET" in ht_test
    assert "correctness_target.name in EVIDENCE_CORRECTNESS_TARGETS" in ht_test
    assert "num_local_ranks == 8" in ht_test
    assert "num_ranks > 8" not in ht_test


def test_final_review_gate_accepts_required_four_gpu_and_compatibility_evidence() -> None:
    decision = validate_final_review_evidence(
        [
            format_correctness_evidence(_args(), result="pass"),
            format_correctness_evidence(
                _args(
                    correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET,
                    correctness_target_label="eight-GPU single-node HT compatibility",
                    ci_correctness_only=False,
                    num_processes=8,
                    num_nodes=1,
                ),
                result="pass",
            ),
            format_correctness_evidence(
                _args(
                    correctness_target=MULTI_NODE_TARGET,
                    correctness_target_label="multi-node HT compatibility",
                    ci_correctness_only=False,
                    num_processes=8,
                    num_nodes=2,
                ),
                result="pass",
            ),
        ],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("pass", "none"),
            MULTI_NODE_TARGET: ("pass", "none"),
        },
    )

    assert decision.accepted is True
    assert decision.blockers == ()


def test_final_review_gate_blocks_missing_ambiguous_and_changed_compatibility_evidence() -> None:
    four_gpu_pass = format_correctness_evidence(_args(), result="pass")
    eight_gpu_pass = format_correctness_evidence(
        _args(
            correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET,
            correctness_target_label="eight-GPU single-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=1,
        ),
        result="pass",
    )
    multi_node_failed = format_correctness_evidence(
        _args(
            correctness_target=MULTI_NODE_TARGET,
            correctness_target_label="multi-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=2,
        ),
        result="fail",
        failure_category="correctness_failed",
    )

    missing = validate_final_review_evidence(
        [four_gpu_pass, eight_gpu_pass],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("pass", "none"),
            MULTI_NODE_TARGET: ("pass", "none"),
        },
    )
    ambiguous = validate_final_review_evidence(
        [four_gpu_pass, eight_gpu_pass, eight_gpu_pass, multi_node_failed],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("pass", "none"),
            MULTI_NODE_TARGET: ("fail", "correctness_failed"),
        },
    )
    changed = validate_final_review_evidence(
        [four_gpu_pass, eight_gpu_pass, multi_node_failed],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("pass", "none"),
            MULTI_NODE_TARGET: ("pass", "none"),
        },
    )
    missing_baseline = validate_final_review_evidence(
        [four_gpu_pass, eight_gpu_pass, multi_node_failed],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("pass", "none"),
        },
    )

    assert missing.accepted is False
    assert "missing_ep_ht_multi_node_compatibility" in missing.blockers
    assert ambiguous.accepted is False
    assert "ambiguous_ep_ht_eight_gpu_single_node_compatibility" in ambiguous.blockers
    assert changed.accepted is False
    assert (
        "changed_ep_ht_multi_node_compatibility_expected_pass_none_got_fail_correctness_failed"
        in changed.blockers
    )
    assert missing_baseline.accepted is False
    assert "missing_expected_compatibility_multi_node" in missing_baseline.blockers


def test_final_review_gate_blocks_infrastructure_limited_compatibility_evidence() -> None:
    four_gpu_pass = format_correctness_evidence(_args(), result="pass")
    blocked_eight_gpu = format_correctness_evidence(
        _args(
            correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET,
            correctness_target_label="eight-GPU single-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=1,
        ),
        result="fail",
        failure_category="gpu_unavailable",
    )
    multi_node_pass = format_correctness_evidence(
        _args(
            correctness_target=MULTI_NODE_TARGET,
            correctness_target_label="multi-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=2,
        ),
        result="pass",
    )

    decision = validate_final_review_evidence(
        [four_gpu_pass, blocked_eight_gpu, multi_node_pass],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("fail", "gpu_unavailable"),
            MULTI_NODE_TARGET: ("pass", "none"),
        },
    )

    assert decision.accepted is False
    assert (
        "blocked_ep_ht_eight_gpu_single_node_compatibility_gpu_unavailable"
        in decision.blockers
    )


def test_final_review_gate_rejects_mislabeled_eight_gpu_compatibility_record() -> None:
    four_gpu_pass = format_correctness_evidence(_args(), result="pass")
    mislabeled_eight_gpu = format_correctness_evidence(
        _args(
            correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET,
            correctness_target_label="eight-GPU single-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=1,
        ),
        result="pass",
    ).replace(
        f"target={EIGHT_GPU_SINGLE_NODE_TARGET}",
        f"target={FOUR_GPU_SINGLE_NODE_TARGET}",
    )
    multi_node_pass = format_correctness_evidence(
        _args(
            correctness_target=MULTI_NODE_TARGET,
            correctness_target_label="multi-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=2,
        ),
        result="pass",
    )

    decision = validate_final_review_evidence(
        [four_gpu_pass, mislabeled_eight_gpu, multi_node_pass],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("pass", "none"),
            MULTI_NODE_TARGET: ("pass", "none"),
        },
    )

    assert decision.accepted is False
    assert (
        "mismatched_ep_ht_eight_gpu_single_node_compatibility_target_expected_"
        "eight_gpu_single_node_got_four_gpu_single_node"
    ) in decision.blockers


def test_final_review_gate_rejects_single_node_shaped_multi_node_record() -> None:
    four_gpu_pass = format_correctness_evidence(_args(), result="pass")
    eight_gpu_pass = format_correctness_evidence(
        _args(
            correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET,
            correctness_target_label="eight-GPU single-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=1,
        ),
        result="pass",
    )
    single_node_shaped_multi_node = format_correctness_evidence(
        _args(
            correctness_target=MULTI_NODE_TARGET,
            correctness_target_label="multi-node HT compatibility",
            ci_correctness_only=False,
            num_processes=8,
            num_nodes=1,
        ),
        result="pass",
    )

    decision = validate_final_review_evidence(
        [four_gpu_pass, eight_gpu_pass, single_node_shaped_multi_node],
        expected_compatibility={
            EIGHT_GPU_SINGLE_NODE_TARGET: ("pass", "none"),
            MULTI_NODE_TARGET: ("pass", "none"),
        },
    )

    assert decision.accepted is False
    assert (
        "mismatched_ep_ht_multi_node_compatibility_num_rdma_ranks_expected_gt_1_got_1"
    ) in decision.blockers


def test_t8_report_tracks_preflight_drift_as_addressed_and_blocks_only_on_runtime_records() -> None:
    issue_dir = REPO_ROOT / "docs" / "issues" / "001-support-ep-ht-4-gpu-single-nod"
    report = (issue_dir / "verify-report.md").read_text(encoding="utf-8")
    verification = (issue_dir / "verification-t-8.md").read_text(encoding="utf-8")

    stale_preflight_claims = (
        "preflight requirements are not fully represented",
        "I did not find explicit GPU-count or IPC/NVLink peer-access preflight checks",
        "Add explicit four-GPU preflight checks",
        "Architectural findings require a decisions.md 'Approved Deviations' entry",
    )
    for stale_claim in stale_preflight_claims:
        assert stale_claim not in report
        assert stale_claim not in verification

    assert "arch_check: PASS" in report
    assert "OVERALL: FAIL" in report
    assert "code-side preflight drift is addressed" in report
    assert "runtime evidence unavailable" in report
    assert "only pending actual four/eight/multi-node runtime records" in verification
