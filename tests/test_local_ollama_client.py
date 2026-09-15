from services.local_ollama_client import build_constrained_chat_payload


def test_constrained_local_chat_payload_disables_thinking_and_bounds_context():
    payload = build_constrained_chat_payload(
        model="qwen3.5:9b-ctx64k",
        prompt="בדיקה",
    )

    assert payload["model"] == "qwen3.5:9b-ctx64k"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["messages"] == [{"role": "user", "content": "בדיקה"}]
    assert payload["options"]["num_ctx"] == 8192
    assert payload["options"]["num_predict"] == 400
    assert payload["options"]["temperature"] == 0.0
