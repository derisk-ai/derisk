"""Bailing Embeddings module."""

from dataclasses import dataclass, field
from typing import Any, List, Type

from derisk._private.pydantic import BaseModel, ConfigDict, Field
from derisk.core import Embeddings
from derisk.core.awel.flow import Parameter, ResourceCategory, register_resource
from derisk.model.adapter.base import register_embedding_adapter
from derisk.model.adapter.embed_metadata import EMBED_COMMON_HF_BAILING_MODELS
from derisk.rag.embedding.embeddings import (
    OpenAPIEmbeddingDeployModelParameters,
    _handle_request_result,
)
from derisk.util.i18n_utils import _


@dataclass
class BailingEmbeddingsDeployModelParameters(OpenAPIEmbeddingDeployModelParameters):
    """Bailing AI Embeddings deploy model parameters."""

    provider: str = "proxy/bailing"
    api_url: str = field(
        default="https://antchat.alipay.com/v1/embeddings",
        metadata={
            "description": _("The URL of the embeddings API."),
            "optional": True,
        },
    )
    backend: str = field(
        default="bge_m3",
        metadata={
            "description": _("The name of the model to use for text embeddings."),
            "optional": True,
        },
    )


@register_resource(
    _("bailing AI Embeddings"),
    "bailing_embeddings",
    category=ResourceCategory.EMBEDDINGS,
    description=_("bailing AI embeddings."),
    parameters=[
        Parameter.build_from(
            _("API Key"),
            "api_key",
            str,
            description=_("Your API key for the bailing AI API."),
        ),
        Parameter.build_from(
            _("Model Name"),
            "model_name",
            str,
            optional=True,
            default="bge_m3",
            description=_("The name of the model to use for text embeddings."),
        ),
    ],
)
class BailingEmbeddings(BaseModel, Embeddings):
    """Bailing AI embeddings.

    This class is used to get embeddings for a list of texts using the Bailing AI API.
    It requires an API key and a model name. The default model name is
    "bailing-embeddings-v2-base-en".
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    api_url: Any  #: :meta private:
    session: Any  #: :meta private:
    api_key: str
    timeout: int = Field(
        default=60, description="The timeout for the request in seconds."
    )
    """API key for the bailing AI API.."""
    model_name: str = "bailing-embeddings-v2-base-en"
    """The name of the model to use for text embeddings. Defaults to
    "bailing-embeddings-v2-base-en"."""

    def __init__(self, **kwargs):
        """Create a new BailingEmbeddings instance."""
        try:
            import requests
        except ImportError:
            raise ValueError(
                "The requests python package is not installed. Please install it with "
                "`pip install requests`"
            )
        if "api_url" not in kwargs:
            kwargs["api_url"] = "https://api.bailing.ai/v1/embeddings"
        if "session" not in kwargs:  # noqa: SIM401
            session = requests.Session()
        else:
            session = kwargs["session"]
        api_key = kwargs.get("api_key")
        if api_key:
            session.headers.update(
                {
                    "Authorization": f"Bearer {api_key}",
                    "Accept-Encoding": "identity",
                }
            )
        kwargs["session"] = session

        super().__init__(**kwargs)

    @classmethod
    def param_class(cls) -> Type[BailingEmbeddingsDeployModelParameters]:
        """Get the parameter class."""
        return BailingEmbeddingsDeployModelParameters

    @classmethod
    def from_parameters(
        cls, parameters: BailingEmbeddingsDeployModelParameters
    ) -> "Embeddings":
        """Create an embedding model from parameters."""
        return cls(
            api_url=parameters.api_url,
            api_key=parameters.api_key,
            model_name=parameters.real_provider_model_name,
            timeout=parameters.timeout,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.
        """
        # Call Bailing AI Embedding API
        # Bailing AI embedding API has a limit of 8192 characters per text
        texts = [text[:8192] for text in texts]
        resp = self.session.post(  # type: ignore
            self.api_url,
            json={"input": texts, "model": self.model_name},
            timeout=self.timeout,
        )
        return _handle_request_result(resp)

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Bailing AI embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


register_embedding_adapter(
    BailingEmbeddings, supported_models=EMBED_COMMON_HF_BAILING_MODELS
)
