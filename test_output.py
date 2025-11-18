#!/usr/bin/env python3
"""Quick Pontus-X test script that writes a dummy artifact to /outputs."""
from __future__ import annotations

import json
import os
import tarfile
from datetime import datetime
from pathlib import Path


def main() -> None:
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/outputs"))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Fall back to /outputs or local artifacts if /data/outputs is read-only.
        for fallback in (Path("/outputs"), Path("artifacts")):
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                output_dir = fallback
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Unable to create any writable output directory")

    payload = {
        "message": "Pontus-X quick test",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    metrics_path = output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    archive_path = output_dir / "test_artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(metrics_path, arcname=metrics_path.name)

    print(f"Wrote {metrics_path} and archived to {archive_path}")


if __name__ == "__main__":
    main()
