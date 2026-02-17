from __future__ import annotations

import os

from openai import OpenAI


class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

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
