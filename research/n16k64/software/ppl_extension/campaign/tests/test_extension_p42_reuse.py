from campaign.extension_p42_reuse import normalize_eval_command


def test_normalize_eval_command_changes_only_plan_path():
    left = ["-m", "campaign.evaluate_ppl", "--model", "m", "--plan", "/a",
            "--domains", "wiki,c4"]
    right = ["-m", "campaign.evaluate_ppl", "--model", "m", "--plan", "/b",
             "--domains", "wiki,c4"]
    assert normalize_eval_command(left) == normalize_eval_command(right)
    assert left[5] == "/a" and right[5] == "/b"


def test_normalize_eval_command_preserves_scientific_options():
    base = ["-m", "campaign.evaluate_ppl", "--model", "m", "--plan", "/a",
            "--length", "2048"]
    changed = base[:-1] + ["1024"]
    assert normalize_eval_command(base) != normalize_eval_command(changed)
