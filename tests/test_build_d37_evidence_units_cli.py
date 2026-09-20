from pathlib import Path
import subprocess
import sys


def test_d37_evidence_builder_cli_loads_project_services_from_the_repo_root():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/build_d37_evidence_units.py", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--annotations" in result.stdout
