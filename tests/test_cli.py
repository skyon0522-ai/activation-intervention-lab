import json

import pytest

from activation_intervention_lab import cli


@pytest.mark.parametrize("kind", ["success", "failure", "empty", "file"])
def test_existing_output_is_preserved_before_experiment(monkeypatch, tmp_path, kind):
    from activation_intervention_lab import experiment
    output = tmp_path / "previous-run"
    if kind == "file":
        output.write_bytes(b"existing file")
        files = [output]
    else:
        output.mkdir()
        names = {"success": ["report.json", "report.html"], "failure": ["failure.json"], "empty": []}[kind]
        files = [output / name for name in names]
        for file in files:
            file.write_bytes(b"prior evidence")
    before = {file: file.read_bytes() for file in files}
    def unexpected_run(**kwargs):
        raise AssertionError("existing output must be refused before model execution")
    monkeypatch.setattr(experiment, "run_experiment", unexpected_run)
    monkeypatch.setattr("sys.argv", ["activation-lab", "--output", str(output)])
    with pytest.raises(SystemExit) as rejected:
        cli.main()
    assert rejected.value.code == 2
    assert {file: file.read_bytes() for file in files} == before
    if output.is_dir():
        assert set(output.iterdir()) == set(files)


def test_new_failed_run_keeps_its_failure_receipt(monkeypatch, tmp_path):
    from activation_intervention_lab import experiment
    output = tmp_path / "failed-run"
    def fail(**kwargs):
        raise RuntimeError("controlled failure")
    monkeypatch.setattr(experiment, "run_experiment", fail)
    monkeypatch.setattr("sys.argv", ["activation-lab", "--output", str(output)])
    with pytest.raises(RuntimeError, match="controlled failure"):
        cli.main()
    assert [p.name for p in output.iterdir()] == ["failure.json"]
    assert json.loads((output / "failure.json").read_text())["failure"]["message"] == "controlled failure"
