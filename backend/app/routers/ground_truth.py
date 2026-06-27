"""
Ground Truth Router — CRUD for ground truth Q&A pairs.

Allows admins to view, add, edit, and delete ground truth entries
without restarting the server.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models import User
from app.schemas import GroundTruthEntry, GroundTruthListResponse
from app.services.auth_service import require_admin
from app.services.ml_service import MLService, get_ml_service
from ml_pipeline.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ground-truth", tags=["ground-truth"])


def _read_ground_truth() -> list[dict]:
    """Read ground truth from the JSON file."""
    gt_path = config.ground_truth_path
    if not gt_path.exists():
        return []
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_ground_truth(entries: list[dict]) -> None:
    """Write ground truth entries to the JSON file."""
    gt_path = config.ground_truth_path
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


@router.get("", response_model=GroundTruthListResponse)
async def list_ground_truth():
    """Get all ground truth Q&A pairs."""
    entries = _read_ground_truth()
    return GroundTruthListResponse(
        entries=[GroundTruthEntry(**e) for e in entries],
        total=len(entries),
    )


@router.post("", response_model=GroundTruthEntry)
async def add_ground_truth(
    entry: GroundTruthEntry,
    ml: MLService = Depends(get_ml_service),
    admin: User = Depends(require_admin),
):
    """Add a new ground truth Q&A pair. Admin only."""
    entries = _read_ground_truth()
    entries.append({"query": entry.query, "answer": entry.answer})
    _write_ground_truth(entries)

    # Reload embeddings in the evaluation engine
    ml._evaluation.reload_ground_truth()

    logger.info("Ground truth entry added by %s: %.50s...", admin.username, entry.query)
    return entry


@router.put("/{index}", response_model=GroundTruthEntry)
async def update_ground_truth(
    index: int,
    entry: GroundTruthEntry,
    ml: MLService = Depends(get_ml_service),
    admin: User = Depends(require_admin),
):
    """Update a ground truth entry by index. Admin only."""
    entries = _read_ground_truth()

    if index < 0 or index >= len(entries):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range")

    entries[index] = {"query": entry.query, "answer": entry.answer}
    _write_ground_truth(entries)

    ml._evaluation.reload_ground_truth()

    logger.info("Ground truth entry %d updated by %s", index, admin.username)
    return entry


@router.delete("/{index}")
async def delete_ground_truth(
    index: int,
    ml: MLService = Depends(get_ml_service),
    admin: User = Depends(require_admin),
):
    """Delete a ground truth entry by index. Admin only."""
    entries = _read_ground_truth()

    if index < 0 or index >= len(entries):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range")

    removed = entries.pop(index)
    _write_ground_truth(entries)

    ml._evaluation.reload_ground_truth()

    logger.info("Ground truth entry %d deleted by %s: %.50s", index, admin.username, removed["query"])
    return {"success": True, "message": f"Entry {index} deleted", "removed": removed}
