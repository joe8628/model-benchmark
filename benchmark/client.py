from typing import Protocol
import anthropic
import openai


class ModelClient(Protocol):
    def complete(
        self,
        prompt: str,
        system: str,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        """Returns (response_text, prompt_tokens, completion_tokens)."""
        ...


class AnthropicClient:
    def __init__(self, model: str):
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        return text, msg.usage.input_tokens, msg.usage.output_tokens


class OpenAIClient:
    def __init__(self, model: str):
        self._model = model
        self._client: openai.OpenAI | None = None

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def complete(self, prompt: str, system: str, temperature: float = 0.0) -> tuple[str, int, int]:
        resp = self._get_client().chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
