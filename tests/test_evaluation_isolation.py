"""Offline evaluation must never consult deployment settings or credentials.

Run: .venv/Scripts/python.exe tests/test_evaluation_isolation.py
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def test_offline_bootstrap_ignores_deployment_sources():
    with tempfile.TemporaryDirectory() as directory:
        environment = dict(os.environ, NA_MESTE_EVALUATION_ROOT=directory,
                           NA_MESTE_EVALUATION_DEVICE="cpu", AI_DEVICE="cuda",
                           DATABASE_URL="sqlite:///must-not-use.db",
                           TELEGRAM_BOT_TOKEN="must-not-load", TELEGRAM_ENABLED="true")
        code = '''
from pathlib import Path
from unittest.mock import patch
import os
root = Path(os.environ['NA_MESTE_EVALUATION_ROOT']).resolve()
with patch('app.operator_settings.read_document', side_effect=AssertionError('deployment JSON read')), \\
     patch('pydantic_settings.sources.DotEnvSettingsSource._read_env_file', side_effect=AssertionError('dotenv read')):
    from app.config import settings
assert settings.database_url == f'sqlite:///{(root / "evaluation.db").as_posix()}'
assert settings.recording_root == root / 'media'
assert settings.topology_config == str(root / 'topology.json')
assert settings.ai_device == 'cpu'
assert settings.telegram_enabled is False
assert settings.telegram_bot_token == ''
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env=environment, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_offline_root_never_falls_back_to_deployment():
    with tempfile.TemporaryDirectory() as directory:
        environment = dict(os.environ, NA_MESTE_EVALUATION_ROOT=str(Path(directory) / "missing"))
        code = '''
from app.operator_settings import OverrideError
try:
    import app.config
except OverrideError as exc:
    assert str(exc) == 'EVALUATION_ROOT_INVALID'
else:
    raise AssertionError('Missing isolated root accepted')
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env=environment, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    tests = [test_offline_bootstrap_ignores_deployment_sources,
             test_invalid_offline_root_never_falls_back_to_deployment]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            import traceback
            traceback.print_exc()
            failures += 1
    print(f"{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(bool(failures))
