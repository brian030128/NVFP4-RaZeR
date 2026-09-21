import json

from campaign import extension_breadth_analysis as A
from campaign import extension_breadth_gates as G


def test_selected_weight_accounting():
    assert A.selected_weights({"selected_tiles": 7, "type_block": [16, 64]}) == 7168
    assert A.selected_weights({"selected_tiles": 14, "type_block": [8, 64]}) == 7168
    assert A.selected_weights({"selected_tiles": None, "type_block": None}) is None


def test_p21_dynamic_jobs_are_bounded_and_serial(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    (root / "queue/jobs").mkdir(parents=True)
    monkeypatch.setattr(G, "CR", root)
    generated = G.stage_jobs("p21", need_selector=True)
    assert len(generated) == 11
    specs = {path.stem: json.loads(path.read_text()) for path in generated}
    assert specs["P21_selector_stats_qwen27b"]["gpus"] == 2
    assert specs["P21_selector_stats_qwen27b"]["reserve"] is True
    assert "--max-memory-gib" in specs["P21_selector_stats_qwen27b"]["command"]
    assert max(row["gpus"] for row in specs.values()) == 2
    assert specs["P21_maps_phi4"]["depends_on"][0] == "P21_maps_qwen27b"
    assert specs["P21_maps_olmo2_13b"]["depends_on"][0] == "P21_maps_phi4"
    assert specs["P21_quality_phi4"]["depends_on"] == ["P21_quality_qwen27b"]
    assert specs["P21_quality_olmo2_13b"]["depends_on"] == ["P21_quality_phi4"]
    assert specs["P21_analysis"]["depends_on"] == ["P21_quality_olmo2_13b"]
    assert specs["P21_terminal"]["depends_on"] == ["P21_analysis"]


def test_p52_reuses_existing_selector_jobs(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    jobs = root / "queue/jobs"
    jobs.mkdir(parents=True)
    for model in G.MODELS:
        (jobs / f"P21_selector_stats_{model}.json").write_text("{}\n")
    monkeypatch.setattr(G, "CR", root)
    generated = G.stage_jobs("p52", need_selector=True)
    assert len(generated) == 8
    assert not any("selector_stats" in path.name for path in generated)
    first = json.loads((jobs / "P52_maps_qwen27b.json").read_text())
    assert "P21_selector_stats_qwen27b" in first["depends_on"]
    assert json.loads((jobs / "P52_terminal.json").read_text())["depends_on"] == ["P52_analysis"]


def test_quality_plan_names_and_resources(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    monkeypatch.setattr(G, "CR", root)
    qwen = G.quality_command("p42", "qwen27b")
    assert str(root / "plans/P42_aggregation_breadth_qwen27b.json") in qwen
    assert qwen[-2:] == ["--max-memory-gib", "44"]
    phi = G.quality_command("p52", "phi4")
    assert str(root / "plans/P52_selector_breadth_phi4.json") in phi
    assert "--max-memory-gib" not in phi
