def _row(model, estimate):
    return {"model": model, "estimate": estimate}


def test_g5_requires_two_model_means_and_all_cell_safety():
    from campaign.extension_p31_analysis import g5_decision

    rows = [_row("llama8b", -0.0006), _row("llama8b", -0.0005),
            _row("qwen4b", -0.0007), _row("qwen4b", -0.0004),
            _row("mistral7b", 0.0002), _row("mistral7b", 0.0003)]
    assert g5_decision(rows)["continue_to_p32"]
    rows[-1]["estimate"] = 0.00101
    assert not g5_decision(rows)["continue_to_p32"]
    rows[-1]["estimate"] = 0.0003
    rows[2]["estimate"] = -0.0002
    rows[3]["estimate"] = -0.0002
    assert not g5_decision(rows)["continue_to_p32"]
