# app/models_llm.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class LLMChatRequest(BaseModel):
    system: Optional[str] = Field(default=None, description="Optional system prompt")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat history")
    user: Optional[str] = Field(default=None, description="Convenience: single user message (appends to messages)")

    # Optional: ask for JSON-structured output
    json_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON Schema for structured response")

    # Optional LLM knobs
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)


class LLMChatResponse(BaseModel):
    ok: bool = True
    model: str
    provider: str
    output_text: Optional[str] = None
    output_json: Optional[Dict[str, Any]] = None
