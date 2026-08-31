from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_local_challenger_preflight_selects_working_python_311_venv():
    result = subprocess.run(
        ["bash", "scripts/run_local_challenger.sh", "--preflight"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Python 3.11" in result.stdout
    assert "PyMuPDF: OK" in result.stdout
    assert "OpenAI API key" not in result.stdout
