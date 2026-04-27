"""Pydantic schemas dùng chung cho agent."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PatientContext(BaseModel):
    age: Optional[int] = Field(None, ge=0, le=130)
    sex: Optional[Literal["male", "female", "other"]] = None
    symptoms: Optional[str] = Field(None, max_length=500)
    history: Optional[str] = Field(None, max_length=1000)


class ClassifyResult(BaseModel):
    mode: str
    labels: list[str]
    scores: list[float]
    top_label: str
    top_score: float
    findings: list[dict] = []
    uncertainty: Optional[dict] = None


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    url: str


class AgentEvent(BaseModel):
    """Sự kiện streaming gửi cho frontend."""

    step: str
    status: Literal["started", "delta", "done", "error"]
    data: Optional[dict] = None
    message: Optional[str] = None
