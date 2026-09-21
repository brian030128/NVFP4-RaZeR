from campaign.extension_p32_gate import resolution_reason, specs


def test_authorized_p32_dag_is_serial_and_complete():
    rows = specs()
    assert [x["job_id"] for x in rows] == [
        "P32_calib_llama8b", "P32_calib_qwen4b", "P32_calib_mistral7b",
        "P32_plan_freeze", "P32_quality_llama8b", "P32_quality_qwen4b",
        "P32_quality_mistral7b", "P32_analysis", "P32_terminal"]
    assert rows[0]["depends_on"] == ["P32_gate"]
    assert all(rows[i]["depends_on"] == [rows[i - 1]["job_id"]]
               for i in range(1, len(rows)))
    assert max(x["gpus"] for x in rows) == 1


def test_stopped_p32_resolution_does_not_claim_the_gate_passed():
    assert "did not meet" in resolution_reason(False)
    assert " met " in resolution_reason(True)
