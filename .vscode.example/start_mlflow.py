import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import urllib.request


class MlflowServerLauncher:
    """Starts the project's local MLflow tracking server for VS Code.

    Used by the "MLflow: start server" task. The address comes from the
    ``tracking`` section of ``configs/train.json``, so the server, the
    training script and the inference scripts always agree. The storage is
    the one documented in the README: ``mlflow.db`` and ``mlartifacts/`` in
    the project root.

    Two lines are printed for the task's readiness detection:
    ``MLFLOW STARTING`` first, then ``MLFLOW READY`` once the health check
    answers (or ``MLFLOW ALREADY RUNNING`` when a server is already there,
    in which case nothing is started).

    Attributes:
        project_root: Folder holding ``configs/``, ``mlflow.db`` and ``mlartifacts/``.
        tracking_uri: Address the server must listen on.
    """

    READY_TIMEOUT_SECONDS = 120

    def __init__(self, project_root: Path) -> None:
        """Initialises the launcher.

        Args:
            project_root: Folder holding ``configs/train.json``.
        """
        self.project_root = project_root
        config = json.loads((project_root / "configs" / "train.json").read_text(encoding="utf-8"))
        self.tracking_uri = config["tracking"]["tracking_uri"]

    def is_healthy(self) -> bool:
        """Checks whether a tracking server answers at the configured address.

        Returns:
            ``True`` when the health endpoint replies ``OK``.
        """
        try:
            with urllib.request.urlopen(f"{self.tracking_uri}/health", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def run(self) -> int:
        """Starts the server and stays attached to it until it stops.

        Returns:
            The server's exit code, or ``0`` when one was already running.
        """
        print("MLFLOW STARTING", flush=True)
        if self.is_healthy():
            print(f"MLFLOW ALREADY RUNNING at {self.tracking_uri}", flush=True)
            return 0
        address = urlparse(self.tracking_uri)
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            str(address.hostname),
            "--port",
            str(address.port),
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--artifacts-destination",
            "./mlartifacts",
        ]
        server = subprocess.Popen(command, cwd=self.project_root)
        threading.Thread(target=self._announce_ready, args=(server,), daemon=True).start()
        try:
            return server.wait()
        except KeyboardInterrupt:
            server.terminate()
            return server.wait()

    def _announce_ready(self, server: subprocess.Popen) -> None:
        """Prints the readiness line once the server answers.

        Args:
            server: The running server process.
        """
        deadline = time.monotonic() + self.READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline and server.poll() is None:
            if self.is_healthy():
                print(f"MLFLOW READY at {self.tracking_uri}", flush=True)
                return
            time.sleep(1)
        print("MLFLOW FAILED to become ready", flush=True)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    sys.exit(MlflowServerLauncher(root.resolve()).run())
