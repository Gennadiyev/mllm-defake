from abc import ABC, abstractmethod

import httpx
from loguru import logger
from openai import OpenAI

from mllm_defake.utils import encode_image_to_base64


class VLLM(ABC):
    def __init__(self):
        self.short_name = "unknown"

    @abstractmethod
    def infer(self, system_prompt: str, user_prompt: str, image_path: str) -> str:
        """
        This method sends the user prompt and image files (if any) to the Vision-LLM API and
        returns the model's response.

        @param system_prompt: A system prompt that gives context to the model.
        @param user_prompt: The question or task prompt provided by the user.
        @param image_path: A path-like object representing the image file to be sent to the API.

        @return: The model's response to the user prompt. Always a string.
        """

    @abstractmethod
    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        """
        This method sends a list of messages to the Vision-LLM API and returns the model's response. The definition of the actual payload may vary from model to model, with behaviors defined within the VLLM implementation.

        @param messages: The payload to be sent to the Vision-LLM API, typically a list of dictionaries if using the OpenAI-compatible interface.

        @return: The model's response to the messages. Always a string.
        """


class Qwen2VLAPI(VLLM):
    def __init__(
        self,
        api_key: str,
        url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        proxy: str | None = None,
    ):
        super().__init__()
        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(api_key=api_key, http_client=self.http_client, base_url=url)
        self.short_name = "qwen2vl"

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model="qwen-vl-max",
            messages=messages,
            max_tokens=1000,
            timeout=120,
        )
        return response.choices[0].message.content

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        max_tokens: int = 700,
    ) -> str:
        image_url = encode_image_to_base64(image_path)
        response = self.client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class OpenAICompat(VLLM):
    def __init__(
        self,
        api_key: str | None = "mllm_defake_key",
        proxy: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the OpenAICompat class of Vision-LLM models. Provide the API key and optionally a proxy URL. Will populate self.api_key, self.http_client, and self.client.

        This class is meant to be used in conjunction with a self-hosted VLLM server instance. By default, it will automatically select the first model available on the server, as VLLM may use folder names with trailing slashes as model names. However, this can be overridden by setting the model_name attribute after initializing this class (as is done in the GPT4o and GPT4oMini classes).

        @param api_key: The API key for the VLLM server.
        @param proxy: The proxy URL to use for the HTTP client, defaults to None, which uses the default OpenAI httpx client.
        @param base_url: The base URL for the VLLM server. If not provided, the default value of None will be used, and will likely send the requests to OpenAI servers.

        @return: None
        """
        super().__init__()
        self.api_key = api_key
        self.http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.client = OpenAI(api_key=api_key, http_client=self.http_client, base_url=base_url)

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=1000, timeout=120
        )
        return response.choices[0].message.content

    def infer(self, system_prompt: str, user_prompt: str, image_path: str) -> str:
        image_url = encode_image_to_base64(image_path)
        response = self.infer_raw(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
        )
        return response

    def query_model_name(self) -> str:
        # Get the model name from the API
        response = self.client.models.list()
        return response.data[0].id


class GPT4o(OpenAICompat):
    """
    Implementation of GPT-4o Vision model.
    Uses the OpenAI-compatible interface for inference with images.
    """

    def __init__(self, api_key: str, proxy: str | None = None):
        # Initialize the parent class with the API key and proxy
        super().__init__(api_key, proxy)
        # Set the specific model name for GPT-4o
        self.model_name = "gpt-4o"
        self.short_name = "gpt4o"


class GPT4oMini(OpenAICompat):
    """
    Implementation of GPT-4o Mini Vision model.
    Uses the OpenAI-compatible interface for inference with images.
    """

    def __init__(self, api_key: str, proxy: str | None = None):
        # Initialize the parent class with the API key and proxy
        super().__init__(api_key, proxy)
        # Set the specific model name for GPT-4o Mini
        self.model_name = "gpt-4o-mini"
        self.short_name = "gpt4omini"


class GPT45(OpenAICompat):
    def __init__(self, api_key: str, proxy: str | None = None):
        # Initialize the parent class with the API key and proxy
        super().__init__(api_key, proxy)
        # Set the specific model name for GPT-4o Mini
        self.model_name = "gpt-4.5-preview"
        self.short_name = "gpt45"


class Llama32VisionInstruct(OpenAICompat):
    """
    Implementation of Llama 3.2 Vision Instruct model.
    Uses the OpenAI-compatible interface for inference with images.
    """

    def __init__(self, api_key: str, proxy: str | None = None, base_url: str | None = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning("No base URL provided for Llama32VisionInstruct, assuming http://127.0.0.1:8000/v1")
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "llama32vi"


class Llama32VisionCoT(OpenAICompat):
    """
    Implementation of Llama 3.2 Vision Chain-of-Thought model.
    Uses the OpenAI-compatible interface for inference with images.

    Will merge system prompt to user prompt if system prompt is provided, as this model does not support system prompts.
    """

    def __init__(self, api_key: str, proxy: str | None = None, base_url: str | None = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning("No base URL provided for Llama32VisionCoT, assuming http://127.0.0.1:8000/v1")
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "llavacot"

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        # This model does not support system prompts, so we replace it with role `user`.
        if messages[0]["role"] == "system":
            messages[0]["role"] = "user"
            messages[0]["content"] = [{"type": "text", "text": messages[0]["content"]}]
        return super().infer_raw(messages)


class QVQ(OpenAICompat):
    """
    Implementation of QVQ Vision model.
    Uses the OpenAI-compatible interface for inference with images.

    Will merge system prompt to user prompt if system prompt is provided, as this model does not support system prompts.
    """

    def __init__(self, api_key: str, proxy: str | None = None, base_url: str | None = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning("No base URL provided for QVQ, assuming http://127.0.0.1:8000/v1")
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "qvq"

    def infer_raw(self, messages: list[dict[str, str]]) -> str:
        # This model does not support system prompts, so we replace it with role `user`.
        if messages[0]["role"] == "system":
            messages[0]["role"] = "user"
            messages[0]["content"] = [{"type": "text", "text": messages[0]["content"]}]
        return super().infer_raw(messages)


class InternVL25(OpenAICompat):
    """
    Implementation of InternVL 2.5 Vision model.
    Uses the OpenAI-compatible interface for inference with images.
    """

    def __init__(self, api_key: str, proxy: str | None = None, base_url: str | None = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning("No base URL provided for InternVL25, assuming http://127.0.0.1:8000/v1")
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "internvl25"


class LLaVAOneVision(OpenAICompat):
    """
    Implementation of LLaVA-OneVision model.
    Uses the OpenAI-compatible interface for inference with images.
    """

    def __init__(self, api_key: str, proxy: str | None = None, base_url: str | None = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning("No base URL provided for LLaVAOneVision, assuming http://127.0.0.1:8000/v1")
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "onevision"


class Qwen2VL(OpenAICompat):
    """
    Implementation of Qwen 2- & 2.5-VL model.
    Uses the OpenAI-compatible interface for inference with images.
    """

    def __init__(self, api_key: str, proxy: str | None = None, base_url: str | None = None):
        # Set default base URL if none is provided
        if base_url is None:
            logger.warning("No base URL provided for Qwen2VL, assuming http://127.0.0.1:8000/v1")
            base_url = "http://127.0.0.1:8000/v1"

        # Initialize the parent class with the API key, proxy, and base URL
        super().__init__(api_key, proxy, base_url)
        # Query the model name from the API
        self.model_name = self.query_model_name()
        self.short_name = "qwen2vl"
