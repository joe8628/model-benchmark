from benchmark.client import AnthropicClient, OpenAIClient


class _StubClient:
    """Satisfies the ModelClient protocol without hitting any API."""
    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        return "The answer is A.", 50, 10


def test_stub_returns_three_tuple():
    client = _StubClient()
    result = client.complete("test prompt", "")
    assert len(result) == 3

def test_stub_response_is_str():
    client = _StubClient()
    response, _, _ = client.complete("test", "")
    assert isinstance(response, str)

def test_stub_token_counts_are_ints():
    client = _StubClient()
    _, prompt_tokens, completion_tokens = client.complete("test", "")
    assert isinstance(prompt_tokens, int)
    assert isinstance(completion_tokens, int)

def test_stub_temperature_accepted():
    client = _StubClient()
    response, _, _ = client.complete("test", "", temperature=0.0)
    assert response

def test_anthropic_client_instantiates():
    # Does not call the API — just confirms the class is importable and constructable
    client = AnthropicClient("claude-sonnet-4-6")
    assert client._model == "claude-sonnet-4-6"

def test_openai_client_instantiates():
    client = OpenAIClient("gpt-4o")
    assert client._model == "gpt-4o"
