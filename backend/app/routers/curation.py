"""
Curation Router — Admin curation queue API endpoints.

Provides endpoints for the human-in-the-loop curation workflow:
  - View pending items
  - Approve (with optional edits)
  - Reject
  - Queue statistics
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import (
    CurationItem,
    CurationQueueResponse,
    CurationActionRequest,
    CurationActionResponse,
    CurationStatsResponse,
)
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/curation", tags=["curation"])


@router.get("/queue", response_model=CurationQueueResponse)
async def get_curation_queue(
    ml: MLService = Depends(get_ml_service),
):
    """
    Get all pending curation items for admin review.

    Returns items awaiting approval along with queue statistics.
    """
    data = ml.get_curation_queue()
    return CurationQueueResponse(
        items=[CurationItem(**item) for item in data["items"]],
        stats=data["stats"],
    )


@router.get("/all", response_model=CurationQueueResponse)
async def get_all_curation_items(
    ml: MLService = Depends(get_ml_service),
):
    """
    Get all curation items (pending, approved, rejected).

    Useful for viewing full curation history.
    """
    data = ml.get_curation_all()
    return CurationQueueResponse(
        items=[CurationItem(**item) for item in data["items"]],
        stats=data["stats"],
    )


@router.post("/approve/{item_id}", response_model=CurationActionResponse)
async def approve_item(
    item_id: str,
    body: CurationActionRequest = CurationActionRequest(),
    ml: MLService = Depends(get_ml_service),
):
    """
    Approve a curation item and append it to the training dataset.

    Optionally provide an edited correction to override the
    teacher-generated one before it enters the training set.
    """
    result = ml.approve_curation_item(item_id, body.edited_correction)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Curation item '{item_id}' not found.")

    logger.info("Curation item approved via API: id=%s", item_id)

    return CurationActionResponse(
        success=True,
        message=f"Item '{item_id}' approved and added to training dataset.",
        item=CurationItem(**result),
    )


@router.post("/reject/{item_id}", response_model=CurationActionResponse)
async def reject_item(
    item_id: str,
    ml: MLService = Depends(get_ml_service),
):
    """
    Reject a curation item (remove from queue without training).
    """
    result = ml.reject_curation_item(item_id)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Curation item '{item_id}' not found.")

    logger.info("Curation item rejected via API: id=%s", item_id)

    return CurationActionResponse(
        success=True,
        message=f"Item '{item_id}' rejected.",
        item=CurationItem(**result),
    )


@router.post("/edit/{item_id}", response_model=CurationActionResponse)
async def edit_and_approve_item(
    item_id: str,
    body: CurationActionRequest,
    ml: MLService = Depends(get_ml_service),
):
    """
    Edit the correction text and approve the item.

    The edited correction replaces the teacher-generated one in both
    the curation record and the training dataset.
    """
    if not body.edited_correction or not body.edited_correction.strip():
        raise HTTPException(
            status_code=400,
            detail="edited_correction is required for edit action.",
        )

    result = ml.approve_curation_item(item_id, body.edited_correction.strip())

    if result is None:
        raise HTTPException(status_code=404, detail=f"Curation item '{item_id}' not found.")

    logger.info("Curation item edited & approved via API: id=%s", item_id)

    return CurationActionResponse(
        success=True,
        message=f"Item '{item_id}' edited and approved.",
        item=CurationItem(**result),
    )


@router.get("/stats", response_model=CurationStatsResponse)
async def get_curation_stats(
    ml: MLService = Depends(get_ml_service),
):
    """
    Get curation queue statistics (pending, approved, rejected counts).
    """
    stats = ml.get_curation_stats()
    return CurationStatsResponse(**stats)
