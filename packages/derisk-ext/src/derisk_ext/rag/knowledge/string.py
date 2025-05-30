"""String Knowledge."""

from typing import Any, Dict, List, Optional, Union

from derisk.core import Document
from derisk.rag.knowledge.base import ChunkStrategy, Knowledge, KnowledgeType


class StringKnowledge(Knowledge):
    """String Knowledge."""

    def __init__(
        self,
        text: str = "",
        knowledge_type: KnowledgeType = KnowledgeType.TEXT,
        encoding: Optional[str] = "utf-8",
        loader: Optional[Any] = None,
        metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Create String knowledge parameters.

        Args:
            text(str): text
            knowledge_type(KnowledgeType): knowledge type
            encoding(str): encoding
            loader(Any): loader
        """
        super().__init__(
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        self._text = text
        self._encoding = encoding

    def _load(self) -> List[Document]:
        """Load raw text from loader."""
        metadata = {"source": "raw text"}
        if self._metadata:
            metadata.update(self._metadata)  # type: ignore
        docs = [Document(content=self._text, metadata=metadata)]
        return docs

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        return ChunkStrategy.CHUNK_BY_SIZE

    @classmethod
    def type(cls):
        """Return knowledge type."""
        return KnowledgeType.TEXT
