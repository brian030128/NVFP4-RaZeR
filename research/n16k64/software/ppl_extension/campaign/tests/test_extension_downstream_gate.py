from campaign.extension_downstream_gate import common_lmeval, gpu_job


def test_accuracy_and_gsm_model_sets_are_not_interchanged():
    """Freeze the intended asymmetry used by the gate's two job loops.

    The strong-outlier-only model must receive GSM8K, but must not be promoted
    into the eight-task accuracy family when it failed PPL eligibility.
    """
    eligible = ["granite8b"]
    outliers = ["falcon3_10b"]
    gsm_models = list(dict.fromkeys([*eligible, *outliers]))
    assert eligible == ["granite8b"]
    assert gsm_models == ["granite8b", "falcon3_10b"]


def test_downstream_jobs_remain_single_a6000():
    row = gpu_job("x", "P72_ACCURACY", "granite8b", "gate", ["cmd"], 1)
    assert row["gpus"] == 1 and row["gpu_model"] == "a6000"
    assert row["depends_on"] == ["gate"]


def test_gsm_command_contains_frozen_suite_and_batch():
    cmd = common_lmeval("granite8b", "/plan", "gsm8k", 4)
    assert cmd[cmd.index("--suite") + 1] == "gsm8k"
    assert cmd[cmd.index("--batch-size") + 1] == "4"
