from core.coaching_rag_system import CoachingRAGSystem

def test_rag_initialization_and_template_population():
    system = CoachingRAGSystem(
        gemini_api_key="fake-key",  # mock or .env override
        chroma_persist_dir="./test_chroma_db"
    )
    assert system.template_collection.count() >= 4
