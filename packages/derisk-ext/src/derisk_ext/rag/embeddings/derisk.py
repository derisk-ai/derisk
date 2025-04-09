"""derisk Embeddings module."""

import json
from dataclasses import dataclass, field
from typing import Any, List, Type
import requests
from derisk._private.pydantic import BaseModel, ConfigDict, Field
from derisk.core import Embeddings
from derisk.core.awel.flow import Parameter, ResourceCategory, register_resource
from derisk.model.adapter.base import register_embedding_adapter
from derisk.model.adapter.embed_metadata import EMBED_COMMON_HF_DERISK_MODELS
from derisk.rag.embedding.embeddings import (
    OpenAPIEmbeddingDeployModelParameters,
    _handle_request_result,
)
from derisk.util.i18n_utils import _


@dataclass
class DeriskEmbeddingsDeployModelParameters(OpenAPIEmbeddingDeployModelParameters):
    """derisk AI Embeddings deploy model parameters."""

    provider: str = "proxy/derisk"
    api_url: str = field(
        default="https://paiplusinferencepre.alipay.com/inference/"
        "9f124aa59397f2c4_r4_embedding/v1",
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
    _("derisk AI Embeddings"),
    "derisk_embeddings",
    category=ResourceCategory.EMBEDDINGS,
    description=_("Derisk AI embeddings."),
    parameters=[
        Parameter.build_from(
            _("API Key"),
            "api_key",
            str,
            description=_("Your API key for the Derisk AI API."),
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
class DeriskEmbeddings(BaseModel, Embeddings):
    """Derisk AI embeddings.

    This class is used to get embeddings for a list of texts using the Derisk AI API.
    It requires an API key and a model name. The default model name is
    "Derisk-embeddings-v2-base-en".
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    api_url: Any  #: :meta private:
    session: Any  #: :meta private:
    api_key: str
    timeout: int = Field(
        default=60, description="The timeout for the request in seconds."
    )
    """API key for the Derisk AI API.."""
    model_name: str = "derisk-embeddings-v2-base-en"
    """The name of the model to use for text embeddings. Defaults to
    "Derisk-embeddings-v2-base-en"."""

    def __init__(self, **kwargs):
        """Create a new DeriskEmbeddings instance."""
        try:
            import requests
        except ImportError:
            raise ValueError(
                "The requests python package is not installed. Please install it with "
                "`pip install requests`"
            )
        if "api_url" not in kwargs:
            kwargs["api_url"] = (
                "https://paiplusinferencepre.alipay.com/inference/9f124aa59397f2c4_r4_embedding/v1"
            )
        if "session" not in kwargs:  # noqa: SIM401
            session = requests.Session()
        else:
            session = kwargs["session"]
        api_key = kwargs.get("api_key")
        session.headers.update(
            {
                "Content-Type": "application/json",
                "MPS-app-name": "test",
                "MPS-http-version": "1.0",
                "MPS-trace-id": "trace_id",
            }
        )
        kwargs["api_key"] = "derisk"
        kwargs["session"] = session

        super().__init__(**kwargs)

    @classmethod
    def param_class(cls) -> Type[DeriskEmbeddingsDeployModelParameters]:
        """Get the parameter class."""
        return DeriskEmbeddingsDeployModelParameters

    @classmethod
    def from_parameters(
        cls, parameters: DeriskEmbeddingsDeployModelParameters
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
        # Call Derisk AI Embedding API
        input = {
            "model": self.model_name,
            "sents": texts,
        }
        data = {"features": {"query": input}}
        resp = self.session.post(  # type: ignore
            self.api_url,
            json=data,
            timeout=self.timeout,
        )
        return _handle_request_result(resp)

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Derisk AI embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


def _handle_request_result(res: requests.Response) -> List[List[float]]:
    """Parse the result from a request.

    Args:
        res: The response from the request.

    Returns:
        List[List[float]]: The embeddings.

    Raises:
        RuntimeError: If the response is not successful.
    """
    res.raise_for_status()
    resp = res.json()
    if "resultMap" not in resp:
        raise RuntimeError(resp["detail"])
    result = resp["resultMap"]["result"]
    if isinstance(result, str):
        data = [result]
    elif result.get("objectValue"):
        data = result["objectValue"]
    embeddings = []
    for d in data:
        json_str = d.replace("'", '"')
        embeddings.append(json.loads(json_str))
    # Sort resulting embeddings by index
    sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore
    # Return just the embeddings
    return [result["embedding"] for result in sorted_embeddings]


register_embedding_adapter(
    DeriskEmbeddings, supported_models=EMBED_COMMON_HF_DERISK_MODELS
)
