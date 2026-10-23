from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from telco_churn.dataset import PROJECT_ROOT


class ReportWriter:
    """Writes machine-readable reports identically on every OS.

    Files always use UTF-8 and LF line endings, and paths stored inside them
    are project-relative, so a Linux and a Windows run produce the same bytes
    for the same results.
    """

    @staticmethod
    def relative(path: Path) -> str:
        """Expresses a path relative to the project root when possible.

        Args:
            path: Any path.

        Returns:
            A POSIX-style project-relative path, or the absolute POSIX path if
            ``path`` lies outside the project.
        """
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return resolved.as_posix()

    @classmethod
    def to_json_safe(cls, value: Any) -> Any:
        """Converts NumPy scalars, arrays, paths and other objects for JSON.

        Args:
            value: Any value, possibly nested in dicts, lists or tuples.

        Returns:
            An equivalent value made of JSON types; unknown objects become
            their ``repr``.
        """
        if isinstance(value, dict):
            return {str(key): cls.to_json_safe(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [cls.to_json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return cls.to_json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return cls.relative(value)
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return repr(value)

    @classmethod
    def write_json(cls, path: Path, payload: dict[str, Any]) -> Path:
        """Writes a JSON report, creating parent directories.

        Args:
            path: Destination file.
            payload: Report content.

        Returns:
            The written path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cls.to_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path

    @staticmethod
    def write_csv(path: Path, frame: pd.DataFrame) -> Path:
        """Writes a CSV table with LF line endings, creating parent directories.

        Args:
            path: Destination file.
            frame: Table to write (index not included).

        Returns:
            The written path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        return path
