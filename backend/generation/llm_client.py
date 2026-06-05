from __future__ import annotations

import os

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider.lower().strip()

        if self.provider == "ollama":
            resolved_base_url = (
                (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1")
                .rstrip("/")
            )
            resolved_api_key = api_key or os.getenv("OLLAMA_API_KEY") or "ollama"
            self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
            return

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=resolved_api_key) if resolved_api_key else None

    def is_available(self) -> bool:
        return self.client is not None

    def complete(self, prompt: str) -> str:
        if not self.client:
            return ""

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            temperature=0,
        )

        return response.output_text.strip()
