"""Execute actual BAT launchers in isolated Windows projects, never this app/DB."""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@contextmanager
def project(*, venv=True):
    with tempfile.TemporaryDirectory(prefix="na meste & launcher ") as directory:
        root = Path(directory)
        for name in ("start_server.bat", "run_server_task.bat"):
            shutil.copyfile(ROOT/name, root/name)
        if venv:
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root/".venv")],
                           check=True, capture_output=True, timeout=30)
        (root/"app").mkdir()
        (root/"app"/"__init__.py").write_text("")
        (root/"app"/"config.py").write_text("import os\nclass Settings:\n    app_port = int(os.environ.get('APP_PORT','8000'))\nsettings=Settings()\n")
        yield root


def launch(root, name, env=None):
    cmd = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
    return subprocess.Popen(f'"{cmd}" /d /s /c ""{root/name}""', cwd=root,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, "PYTHONIOENCODING":"utf-8", **(env or {})},
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def finish(process):
    if process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10)
    return process.communicate(timeout=10)


@test
def foreground_uses_preferred_then_fallback_venv_and_preserves_exit():
    with project() as root:
        shutil.copytree(root/".venv", root/"venv")
        (root/"main.py").write_text("import sys\nfrom pathlib import Path\nPath('selected.txt').write_text(Path(sys.prefix).name)\nraise SystemExit(23)\n")
        for expected in (".venv", "venv"):
            process = launch(root, "start_server.bat")
            output = process.communicate(timeout=10)
            assert (root/"selected.txt").read_text() == expected, output
            assert process.returncode == 23, (process.returncode, output)
            if expected == ".venv":
                shutil.rmtree(root/".venv")


@test
def missing_venv_fails_both_launchers_without_global_python():
    with project(venv=False) as root:
        for name in ("start_server.bat", "run_server_task.bat"):
            process = launch(root, name)
            output = process.communicate(timeout=10)
            assert process.returncode == 1, (name, process.returncode, output)
            assert not (root/"data").exists()


@contextmanager
def signal_socket():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(8)
        yield server


def fake_app(root):
    (root/"main.py").write_text("import os,socket,time\nfrom pathlib import Path\n"
        "Path('started.txt').write_text('started')\nprint('ISOLATED_APP_STARTED',flush=True)\n"
        "with socket.create_connection(('127.0.0.1',int(os.environ['TEST_SIGNAL_PORT']))) as s: s.sendall(b'started')\n"
        "time.sleep(60)\n")


def receive_signal(server, timeout=8):
    server.settimeout(timeout)
    connection, _ = server.accept()
    with connection:
        return connection.recv(20)


def listed_port(root, *, busy):
    # Real CMD pipeline with a controlled OS listing and observable poll count.
    text = "@echo off\r\necho check>>polls.txt\r\n"
    if busy:
        text += "echo   TCP    0.0.0.0:18791    0.0.0.0:0    LISTENING    123\r\n"
    text += '%PYTHON% -c "import os,socket; s=socket.create_connection((\'127.0.0.1\',int(os.environ[\'TEST_SIGNAL_PORT\']))); s.sendall(b\'poll\'); s.close()"\r\n'
    (root/"netstat.cmd").write_text(text)


@test
def watcher_creates_log_directory_before_first_launch():
    with project() as root, signal_socket() as signal:
        fake_app(root)
        # Controlled OS listing: no occupied app port. No production port probing.
        (root/"netstat.cmd").write_text("@echo off\r\nexit /b 0\r\n")
        env = {"APP_PORT":"18790", "TEST_SIGNAL_PORT":str(signal.getsockname()[1])}
        process = launch(root, "run_server_task.bat", env)
        try:
            try:
                conn, _ = signal.accept()
            except TimeoutError:
                diagnostic = finish(process)
                process = None
                raise AssertionError(("watcher never launched from a fresh project without data/", diagnostic)) from None
            with conn:
                assert conn.recv(20) == b"started"
            assert "ISOLATED_APP_STARTED" in (root/"data"/"app_run.log").read_text(errors="replace")
        finally:
            if process is not None:
                finish(process)


@test
def watcher_respects_configured_port_and_waits_when_busy():
    with project() as root, signal_socket() as signal:
        fake_app(root)
        listed_port(root, busy=True)
        process = launch(root, "run_server_task.bat", {"APP_PORT":"18791", "TEST_SIGNAL_PORT":str(signal.getsockname()[1])})
        try:
            assert receive_signal(signal) == b"poll"
            try:
                repeated = receive_signal(signal, 2)
            except TimeoutError:
                assert not (root/"started.txt").exists()
                assert process.poll() is None, "watcher exited instead of waiting on configured busy port"
                assert (root/"polls.txt").read_text().splitlines() == ["check"]
            else:
                raise AssertionError(("watcher launched an app or busy-polled with redirected stdin", repeated))
        finally:
            finish(process)


@test
def watcher_rejects_invalid_ports_even_when_python_is_optimized():
    with project() as root:
        (root/"main.py").write_text("from pathlib import Path\nimport time\nPath('started.txt').write_text('started')\ntime.sleep(60)\n")
        (root/"netstat.cmd").write_text("@echo off\r\nexit /b 0\r\n")
        for port in ("0", "-1", "65536"):
            process = launch(root, "run_server_task.bat", {"APP_PORT":port, "PYTHONOPTIMIZE":"1"})
            try:
                try:
                    output = process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    raise AssertionError(("invalid port did not exit promptly", port)) from None
                assert process.returncode == 1, (port, process.returncode, output)
                assert not (root/"started.txt").exists()
            finally:
                finish(process)


@test
def watcher_delays_restart_with_redirected_stdin():
    with project() as root, signal_socket() as signal:
        listed_port(root, busy=False)
        (root/"main.py").write_text("import os,socket\nfrom pathlib import Path\n"
            "with Path('attempts.txt').open('a') as f: f.write('start\\n')\n"
            "with socket.create_connection(('127.0.0.1',int(os.environ['TEST_SIGNAL_PORT']))) as s: s.sendall(b'run')\n"
            "raise SystemExit(23)\n")
        process = launch(root, "run_server_task.bat", {"APP_PORT":"18791", "TEST_SIGNAL_PORT":str(signal.getsockname()[1])})
        try:
            assert receive_signal(signal) == b"poll"
            assert receive_signal(signal) == b"run"
            try:
                repeated = receive_signal(signal, 2)
            except TimeoutError:
                assert process.poll() is None, "watcher exited instead of waiting to restart"
                assert (root/"attempts.txt").read_text().splitlines() == ["start"]
                assert (root/"polls.txt").read_text().splitlines() == ["check"]
            else:
                raise AssertionError(("watcher restarted without delay", repeated))
        finally:
            finish(process)


if __name__ == "__main__":
    if os.name != "nt":
        print("Windows-only launcher checks skipped")
        raise SystemExit(0)
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"{len(TESTS)-failed}/{len(TESTS)} passed")
    raise SystemExit(bool(failed))
