import asyncio
import os
from dataclasses import asdict
from typing import Dict, List
import uuid  # Add this line at the top

import chromadb
from backend.schemas.data_models import AudioChunk, ModelInferences, RealTimeFeedback, SessionReport


class ChromaDBStorage:
    """Handles ChromaDB integration for embeddings and metadata storage"""

    def __init__(self, persist_directory: str = None):
        persist_directory = persist_directory or os.getenv("CHROMADB_PERSIST_DIR", "./.chromadb")
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Two collections: one for real-time feedback, one for final reports
        self.sessions_collection = self.client.get_or_create_collection("coaching_sessions")
        self.feedback_collection = self.client.get_or_create_collection("real_time_feedback")

    async def store_session_chunk(
        self, session_id: str, chunk: AudioChunk, inferences: ModelInferences, feedback: RealTimeFeedback
    ):
        """Store processed chunk data with embeddings + metadata"""
        metadata = {
            "session_id": session_id,
            "timestamp": chunk.timestamp,
            "speaker": chunk.speaker,
            "emotion_primary": max(inferences.emotion.items(), key=lambda x: x[1])[0],
            "interest_level": inferences.interest_level,
            "grow_phase": feedback.grow_phase.phase,
            "engagement_score": feedback.engagement_score,
        }

        chunk_id = f"{session_id}_{chunk.timestamp}_{uuid.uuid4().hex[:6]}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.feedback_collection.add(
                documents=[chunk.transcript],
                metadatas=[metadata],
                ids=[chunk_id],
            ),
        )

    async def store_session_report(self, report: SessionReport):
        """Store final session report summary & metadata"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.sessions_collection.add(
                documents=[report.transcript_summary],
                metadatas=[asdict(report)],
                ids=[report.session_id],
            ),
        )

    async def get_session_context(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent session context"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.feedback_collection.query(
                query_texts=["recent context"],
                where={"session_id": session_id},
                n_results=limit,
            ),
        )
        return results.get("metadatas", [[]])[0]