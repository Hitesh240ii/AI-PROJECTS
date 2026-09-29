import pytest

from app.errors import ConfigurationError, KnowledgeBotError, LLMError
from app.llm.groq_client import GroqClient
from app.services.qa_service import NO_ANSWER_MESSAGE, NO_DOCUMENTS_MESSAGE
from tests.conftest import CAFETERIA_TEXT, LEAVE_TEXT


def test_grounded_answer_with_sources(service, llm):
    service.ingest_file("leave.txt", LEAVE_TEXT)
    service.ingest_file("cafeteria.txt", CAFETERIA_TEXT)
    answer = service.ask("How many days of annual leave do employees get?")
    assert answer.llm_used and "20 days" in answer.answer
    assert answer.sources[0].filename == "leave.txt"
    assert answer.sources[0].cited is True
    assert "leave.txt" in answer.sources[0].label
    question, context = llm.calls[0]
    assert "[Source 1]" in context and "annual leave" in context
    assert "cafeteria" not in context  # irrelevant chunk was filtered by the threshold


def test_no_relevant_documents_skips_llm(service, llm):
    service.ingest_file("leave.txt", LEAVE_TEXT)
    answer = service.ask("What is the capital of France?")
    assert answer.answer == NO_ANSWER_MESSAGE
    assert answer.sources == [] and not answer.llm_used
    assert llm.calls == []  # LLM never saw irrelevant context


def test_ask_with_empty_knowledge_base(service, llm):
    assert service.ask("Anything?").answer == NO_DOCUMENTS_MESSAGE
    assert llm.calls == []


def test_empty_question_rejected(service):
    with pytest.raises(KnowledgeBotError):
        service.ask("   ")


def test_duplicate_upload_is_skipped(service, store):
    first = service.ingest_file("leave.txt", LEAVE_TEXT)
    second = service.ingest_file("copy_of_leave.txt", LEAVE_TEXT)
    assert first.status == "indexed" and second.status == "duplicate"
    assert store.count() == first.chunks


def test_bad_inputs_do_not_crash(service):
    assert service.ingest_file("notes.xyz", b"hello").status == "failed"
    assert service.ingest_file("empty.txt", b"").status == "failed"
    assert service.ingest_file("blank.txt", b"   \n ").status == "failed"
    assert service.ingest_file("broken.pdf", b"not really a pdf").status == "failed"
    assert service.ingest_file("broken.docx", b"not really a docx").status == "failed"


def test_one_bad_file_does_not_block_others(service):
    results = service.ingest_files([("bad.pdf", b"junk"), ("leave.txt", LEAVE_TEXT)])
    assert [r.status for r in results] == ["failed", "indexed"]


def test_delete_and_clear(service, store):
    r = service.ingest_file("leave.txt", LEAVE_TEXT)
    assert [d.filename for d in service.list_documents()] == ["leave.txt"]
    service.delete_document(r.document_id)
    assert store.count() == 0
    service.ingest_file("leave.txt", LEAVE_TEXT)
    service.clear_knowledge_base()
    assert store.count() == 0


def test_llm_failure_propagates_as_friendly_error(service, llm):
    service.ingest_file("leave.txt", LEAVE_TEXT)

    def boom(question, context):
        raise LLMError("Groq rate limit reached.")

    llm.generate = boom
    with pytest.raises(LLMError) as info:
        service.ask("annual leave days")
    assert "rate limit" in info.value.user_message


def test_missing_groq_key_gives_configuration_error():
    client = GroqClient(api_key="", model="llama-3.3-70b-versatile")
    assert not client.is_configured
    with pytest.raises(ConfigurationError) as info:
        client.generate("q", "ctx")
    assert "GROQ_API_KEY" in info.value.user_message


def test_groq_client_builds_grounded_prompt():
    captured = {}

    class _Msg:  # minimal stand-in for the Groq SDK response
        content = " Answer [Source 1] "

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    fake = type("F", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()
    client = GroqClient("k", "some-model", client=fake)
    assert client.generate("Q?", "CTX") == "Answer [Source 1]"
    assert captured["model"] == "some-model"
    assert "only" in captured["messages"][0]["content"].lower()
    assert "<context>\nCTX\n</context>" in captured["messages"][1]["content"]
