import json
from asyncio import sleep
from urllib.parse import urlparse

import httpx

from delphi import logger

from .client import Client, Response
from .types import ChatFormatRequest

# Preferred provider routing arguments.
# Change depending on what model you'd like to use.
PROVIDER = {"order": ["Together", "DeepInfra"]}


class OpenRouter(Client):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url="https://openrouter.ai/api/v1/chat/completions",
        max_tokens: int = 3000,
        temperature: float = 1.0,
    ):
        super().__init__(model)

        if not api_key:
            raise ValueError("OpenRouter API key is empty.")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"Using OpenRouter model: {model} with base URL: {base_url}")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.max_tokens = max_tokens
        self.temperature = temperature
        timeout_config = httpx.Timeout(5.0)
        self.client = httpx.AsyncClient(timeout=timeout_config)
        logger.warning("We currently don't support logprobs for OpenRouter")

    def postprocess(self, response):
        response_json = response.json()
        msg = response_json["choices"][0]["message"]["content"]
        return Response(text=msg)

    async def generate(  # type: ignore
        self,
        prompt: ChatFormatRequest,
        max_retries: int = 3,
        **kwargs,  # type: ignore
    ) -> Response:  # type: ignore
        kwargs.pop("schema", None)
        # We have to decide if we want to do this like this or not
        # Currently only simulation uses generation kwargs.
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        data = {
            "model": self.model,
            "messages": prompt,
            # "provider": PROVIDER,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url=self.url, json=data, headers=self.headers, timeout=100
                )
                response.raise_for_status()
                result = self.postprocess(response)

                return result

            except json.JSONDecodeError:
                logger.warning(
                    f"Attempt {attempt + 1}: Invalid JSON response, retrying..."
                )

            except httpx.HTTPStatusError as e:
                body = e.response.text[:1000]
                logger.warning(
                    f"Attempt {attempt + 1}: HTTP {e.response.status_code} "
                    f"from {urlparse(self.url).netloc}: {body}"
                )
                if e.response.status_code in {400, 401, 402, 403, 404}:
                    break

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {repr(e)}, retrying...")

            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")
