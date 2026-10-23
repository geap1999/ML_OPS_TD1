from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from telco_churn.dataset import PROJECT_ROOT


class RunProvenance:
    """Captures the code and environment identity of a run.

    Git is the primary code identity; because a run can happen from an
    uncommitted working tree (or without Git at all), the SHA-256 of every
    package source file is recorded too.

    Attributes:
        PACKAGES: Distributions whose installed versions are recorded.
    """

    PACKAGES = ("telco_churn", "pandas", "numpy", "scikit-learn", "scipy", "joblib")

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        """Initialises the recorder.

        Args:
            project_root: Repository root used for Git and source hashing.
        """
        self.project_root = project_root

    @staticmethod
    def sha256_file(path: Path) -> str:
        """Hashes a file's bytes.

        Args:
            path: File to hash.

        Returns:
            Hex SHA-256 digest.
        """
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def timestamp() -> str:
        """Returns the current UTC time.

        Returns:
            ISO-8601 timestamp.
        """
        return datetime.now(UTC).isoformat()

    def environment(self) -> dict[str, Any]:
        """Records the interpreter, platform and library versions.

        Returns:
            Python version, platform string and package versions.
        """
        return {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in self.PACKAGES},
        }

    def code(self) -> dict[str, Any]:
        """Records the Git commit and the hashes of the package sources.

        Returns:
            Commit hash (or ``None``), whether the tree had uncommitted
            changes (or ``None``), and per-file SHA-256 of ``telco_churn/``.
        """
        sources = sorted((self.project_root / "telco_churn").glob("*.py"))
        return {
            "git_commit": self._git("rev-parse", "HEAD"),
            "git_dirty": self._git_dirty(),
            "source_sha256": {
                path.relative_to(self.project_root).as_posix(): self.sha256_file(path)
                for path in sources
            },
        }

    def _git(self, *args: str) -> str | None:
        """Runs a read-only Git command in the project root.

        Args:
            *args: Git arguments.

        Returns:
            Stripped standard output, or ``None`` when Git is unavailable or
            the command fails.
        """
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    def _git_dirty(self) -> bool | None:
        """Tells whether the working tree has uncommitted changes.

        Returns:
            ``True`` if there are changes, ``False`` if clean, ``None`` if
            unknown.
        """
        status = self._git("status", "--porcelain")
        return None if status is None else bool(status)
