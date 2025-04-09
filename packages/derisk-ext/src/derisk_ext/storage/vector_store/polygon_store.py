"""Polygon vector store."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

from derisk.core import Chunk, Embeddings
from derisk.core.awel.flow import Parameter, ResourceCategory, register_resource
from derisk.storage.vector_store.base import (
    _COMMON_PARAMETERS,
    VectorStoreBase,
    VectorStoreConfig,
)
from derisk.storage.vector_store.filters import FilterOperator, MetadataFilters
from derisk.util import string_utils
from derisk.util.i18n_utils import _

logger = logging.getLogger(__name__)


@register_resource(
    _("Polygon Config"),
    "polygon_vector_config",
    category=ResourceCategory.VECTOR_STORE,
    parameters=[
        *_COMMON_PARAMETERS,
        Parameter.build_from(
            _("Uri"),
            "uri",
            str,
            description=_(
                "The uri of polygon store, if not set, will use the default uri."
            ),
            optional=True,
            default=None,
        ),
        Parameter.build_from(
            _("Port"),
            "port",
            str,
            description=_(
                "The port of polygon store, if not set, will use the default port."
            ),
            optional=True,
            default="19530",
        ),
        Parameter.build_from(
            _("Alias"),
            "alias",
            str,
            description=_(
                "The alias of polygon store, if not set, will use the default alias."
            ),
            optional=True,
            default="default",
        ),
        Parameter.build_from(
            _("Primary Field"),
            "primary_field",
            str,
            description=_(
                "The primary field of polygon store, if not set, will use the "
                "default primary field."
            ),
            optional=True,
            default="pk_id",
        ),
        Parameter.build_from(
            _("Text Field"),
            "text_field",
            str,
            description=_(
                "The text field of polygon store, if not set, will use the "
                "default text field."
            ),
            optional=True,
            default="content",
        ),
        Parameter.build_from(
            _("Embedding Field"),
            "embedding_field",
            str,
            description=_(
                "The embedding field of polygon store, if not set, will use the "
                "default embedding field."
            ),
            optional=True,
            default="vector",
        ),
    ],
    description=_("Polygon vector config."),
)
@dataclass
class PolygonStoreConfig(VectorStoreConfig):
    """PolygonStore vector store config."""

    __type__ = "polygonstore"

    alias: str = field(
        default="default",
        metadata={
            "description": "The alias of polygon store, if not set, "
            "will use the default alias."
        },
    )
    token: str = field(
        default=None,
        metadata={"description": "polygon token"},
    )
    primary_field: str = field(
        default="pk_id",
        metadata={
            "description": "The primary field of polygon store, if not set, "
            "will use the default primary field."
        },
    )
    text_field: str = field(
        default="content",
        metadata={
            "description": "The text field of polygon store, if not set, "
            "will use the default text field."
        },
    )
    embedding_field: str = field(
        default="vector",
        metadata={
            "description": "The embedding field of polygon store, if not set, "
            "will use the default embedding field."
        },
    )
    metadata_field: str = field(
        default="metadata",
        metadata={
            "description": "The metadata field of polygon store, if not set, "
            "will use the default metadata field."
        },
    )
    secure: str = field(
        default="",
        metadata={"description": "The secure of polygon store, if not set, "},
    )

    def create_store(self, **kwargs) -> "PolygonStore":
        """Create a PolygonStore instance."""
        return PolygonStore(vector_store_config=self, **kwargs)


@register_resource(
    _("Polygon Vector Store"),
    "Polygon_vector_store",
    category=ResourceCategory.VECTOR_STORE,
    description=_("Polygon vector store."),
    parameters=[
        Parameter.build_from(
            _("Polygon Config"),
            "vector_store_config",
            PolygonStoreConfig,
            description=_("the Polygon config of vector store."),
            optional=True,
            default=None,
        ),
    ],
)
class PolygonStore(VectorStoreBase):
    """Polygon vector store."""

    def __init__(
        self,
        vector_store_config: PolygonStoreConfig,
        name: Optional[str] = None,
        embedding_fn: Optional[Embeddings] = None,
    ) -> None:
        """Create a PolygonStore instance.

        Args:
            vector_store_config (PolygonVectorConfig): PolygonStore config.
            refer to https://Polygon.io/docs/v2.0.x/manage_connection.md
        """
        super().__init__()
        self._vector_store_config = vector_store_config

        try:
            from polygonmilvus import connections
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )
        # connect_kwargs = {}
        polygon_vector_config = vector_store_config.to_dict()
        self.token = polygon_vector_config.get("token")

        self.collection_name = name
        if string_utils.contains_chinese(self.collection_name):
            bytes_str = self.collection_name.encode("utf-8")
            hex_str = bytes_str.hex()
            self.collection_name = hex_str
        if embedding_fn is None:
            # Perform runtime checks on self.embedding to
            # ensure it has been correctly set and loaded
            raise ValueError("embedding_fn is required for polygonStore")
        self.embedding: Embeddings = embedding_fn
        self.fields: List = []
        self.alias = polygon_vector_config.get("alias") or "default"

        # use HNSW by default.
        self.index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64},
        }

        # use HNSW by default.
        self.index_params_map = {
            "IVF_FLAT": {"params": {"nprobe": 10}},
            "IVF_SQ8": {"params": {"nprobe": 10}},
            "IVF_PQ": {"params": {"nprobe": 10}},
            "HNSW": {"params": {"M": 8, "efConstruction": 64}},
            "RHNSW_FLAT": {"params": {"ef": 10}},
            "RHNSW_SQ": {"params": {"ef": 10}},
            "RHNSW_PQ": {"params": {"ef": 10}},
            "IVF_HNSW": {"params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"params": {"search_k": 10}},
        }
        # default collection schema
        self.primary_field = polygon_vector_config.get("primary_field") or "pk_id"
        self.chunk_id = polygon_vector_config.get("chunk_id") or "chunk_id"
        self.vector_field = polygon_vector_config.get("embedding_field") or "vector"
        self.text_field = polygon_vector_config.get("text_field") or "content"
        self.metadata_field = polygon_vector_config.get("metadata_field") or "metadata"

        connections.connect(token=self.token, alias="default")
        self.init_collection_schema(self.collection_name)

    def init_collection_schema(self, vector_name: str) -> str:
        """Create a polygon collection.

        Create a polygon collection, indexes it with HNSW, load document.

        Args:
            vector_name (str): your collection name.
        Returns:
            str: document collection.
        """
        try:
            from polygonmilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                utility,
            )
            from polygonmilvus.orm.type import infer_dtype_bydata  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )
        # if not connections.has_connection("default"):
        #     connections.connect(
        #         host=self.uri or "127.0.0.1",
        #         port=self.port or "19530",
        #         alias="default",
        #         # secure=self.secure,
        #     )
        embeddings = self.embedding.embed_query("test")

        if utility.has_collection(self.collection_name):
            self.col = Collection(self.collection_name, using=self.alias)
            self.fields = []
            for x in self.col.schema.fields:
                self.fields.append(x.name)
                if x.auto_id:
                    self.fields.remove(x.name)
                if x.is_primary:
                    self.primary_field = x.name
                if (
                    x.dtype == DataType.FLOAT_VECTOR
                    or x.dtype == DataType.BINARY_VECTOR
                ):
                    self.vector_field = x.name
            return vector_name
            # return self.collection_name
        dim = len(embeddings)
        # Generate unique names
        primary_field = self.primary_field
        chunk_id = self.chunk_id
        vector_field = self.vector_field
        text_field = self.text_field
        metadata_field = self.metadata_field
        collection_name = vector_name
        fields = []
        # max_length = 0
        # Create the text field
        fields.append(FieldSchema(text_field, DataType.VARCHAR, max_length=65535))
        # primary key field
        fields.append(
            FieldSchema(primary_field, DataType.VARCHAR, is_primary=True, auto_id=True)
        )
        fields.append(FieldSchema(chunk_id, DataType.VARCHAR, max_length=65535))
        # vector field
        fields.append(FieldSchema(vector_field, DataType.FLOAT_VECTOR, dim=dim))

        fields.append(FieldSchema(metadata_field, DataType.JSON))
        schema = CollectionSchema(fields)
        # Create the collection
        collection = Collection(collection_name, schema)
        self.col = collection
        # index parameters for the collection
        index = self.index_params
        # polygon index
        collection.create_index(vector_field, index)
        # collection.load()
        # schema = collection.schema
        return vector_name

    def init_schema_and_load(self, vector_name, documents) -> List[str]:
        """Create a polygon collection.

        Create a polygon collection, indexes it with HNSW, load document.

        Args:
            vector_name (Embeddings): your collection name.
            documents (List[str]): Text to insert.
        Returns:
            List[str]: document ids.
        """
        try:
            from polygonmilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                utility,
            )
            from polygonmilvus.orm.type import infer_dtype_bydata  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )
        # if not connections.has_connection("default"):
        #     connections.connect(
        #         host=self.uri or "127.0.0.1",
        #         port=self.port or "19530",
        #         alias="default",
        #         # secure=self.secure,
        #     )
        texts = [d.content for d in documents]
        metadatas = [d.metadata for d in documents]
        chunk_ids = [d.chunk_id for d in documents]
        embeddings = self.embedding.embed_query(texts[0])

        if utility.has_collection(self.collection_name):
            self.col = Collection(self.collection_name, using=self.alias)
            self.fields = []
            for x in self.col.schema.fields:
                self.fields.append(x.name)
                if x.auto_id:
                    self.fields.remove(x.name)
                if x.is_primary:
                    self.primary_field = x.name
                if (
                    x.dtype == DataType.FLOAT_VECTOR
                    or x.dtype == DataType.BINARY_VECTOR
                ):
                    self.vector_field = x.name
            return self._add_documents(texts, metadatas, chunk_ids)
            # return self.collection_name
        dim = len(embeddings)
        # Generate unique names
        primary_field = self.primary_field
        chunk_id = self.chunk_id
        vector_field = self.vector_field
        text_field = self.text_field
        metadata_field = self.metadata_field
        collection_name = vector_name
        fields = []
        max_length = 0
        for y in texts:
            max_length = max(max_length, len(y))
        # Create the text field
        fields.append(FieldSchema(text_field, DataType.VARCHAR, max_length=65535))
        # primary key field
        fields.append(
            FieldSchema(primary_field, DataType.VARCHAR, is_primary=True, auto_id=True)
        )
        fields.append(FieldSchema(chunk_id, DataType.VARCHAR, max_length=65535))
        # vector field
        fields.append(FieldSchema(vector_field, DataType.FLOAT_VECTOR, dim=dim))

        fields.append(FieldSchema(metadata_field, DataType.JSON))
        schema = CollectionSchema(fields)
        # Create the collection
        collection = Collection(collection_name, schema)
        self.col = collection
        # index parameters for the collection
        index = self.index_params
        # polygon index
        collection.create_index(vector_field, index)
        collection.load()
        schema = collection.schema
        for x in schema.fields:
            self.fields.append(x.name)
            if x.auto_id:
                self.fields.remove(x.name)
            if x.is_primary:
                self.primary_field = x.name
            if x.dtype == DataType.FLOAT_VECTOR or x.dtype == DataType.BINARY_VECTOR:
                self.vector_field = x.name
        ids = self._add_documents(texts, metadatas, chunk_ids)

        return ids

    def _add_documents(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        chunk_ids: Optional[List[str]] = None,
        partition_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> List[str]:
        """Add text data into polygon."""
        insert_dict: Any = {self.text_field: list(texts), "chunk_id": chunk_ids}
        try:
            import numpy as np  # noqa: F401

            text_vector = self.embedding.embed_documents(list(texts))
            insert_dict[self.vector_field] = text_vector
        except NotImplementedError:
            insert_dict[self.vector_field] = [
                self.embedding.embed_query(x) for x in texts
            ]
        # Collect the metadata into the insert dict.
        # self.fields.extend(metadatas[0].keys())
        if len(self.fields) > 2 and metadatas is not None:
            for d in metadatas:
                # for key, value in d.items():
                insert_dict.setdefault("metadata", []).append(d)
        # Convert dict to list of lists for insertion
        insert_list = [insert_dict[x] for x in self.fields]
        # Insert into the collection.
        res = self.col.insert(
            insert_list, partition_name=partition_name, timeout=timeout
        )
        if res:
            logger.info(
                f"inserted {res.success_count} "
                f"documents into collection {self.col.name}"
            )

        return [str(pk_id) for pk_id in res.primary_keys]

    def get_config(self) -> PolygonStoreConfig:
        """Get the vector store config."""
        return self._vector_store_config

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """Load document in vector database."""
        batch_size = 500
        batched_list = [
            chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
        ]
        doc_ids = []
        for doc_batch in batched_list:
            doc_ids.extend(
                self.init_schema_and_load(
                    vector_name=self.collection_name, documents=doc_batch
                )
            )
        doc_ids = [str(doc_id) for doc_id in doc_ids]
        return doc_ids

    def similar_search(
        self, text, topk, filters: Optional[MetadataFilters] = None
    ) -> List[Chunk]:
        """Perform a search on a query string and return results."""
        try:
            from polygonmilvus import Collection, DataType
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )
        """similar_search in vector database."""
        self.col = Collection(self.collection_name)
        schema = self.col.schema
        for x in schema.fields:
            self.fields.append(x.name)
            if x.auto_id:
                self.fields.remove(x.name)
            if x.is_primary:
                self.primary_field = x.name
            if x.dtype == DataType.FLOAT_VECTOR or x.dtype == DataType.BINARY_VECTOR:
                self.vector_field = x.name
        # convert to milvus expr filter.
        milvus_filter_expr = self.convert_metadata_filters(filters) if filters else None
        _, docs_and_scores = self._search(text, topk, expr=milvus_filter_expr)

        return [
            Chunk(
                metadata=json.loads(doc.metadata.get("metadata", "")),
                content=doc.content,
            )
            for doc, _, _ in docs_and_scores
        ]

    def similar_search_with_scores(
        self,
        text: str,
        topk: int,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Chunk]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            text (str): The query text.
            topk (int): The number of similar documents to return.
            score_threshold (float): Optional, a floating point value between 0 to 1.
            filters (Optional[MetadataFilters]): Optional, metadata filters.
        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        try:
            from polygonmilvus import Collection, DataType
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )

        self.col = Collection(self.collection_name)
        schema = self.col.schema
        for x in schema.fields:
            self.fields.append(x.name)
            if x.auto_id:
                self.fields.remove(x.name)
            if x.is_primary:
                self.primary_field = x.name

            if x.dtype == DataType.FLOAT_VECTOR or x.dtype == DataType.BINARY_VECTOR:
                self.vector_field = x.name
        # convert to milvus expr filter.
        milvus_filter_expr = self.convert_metadata_filters(filters) if filters else None
        _, docs_and_scores = self._search(query=text, k=topk, expr=milvus_filter_expr)
        if any(score < 0.0 or score > 1.0 for _, score, id in docs_and_scores):
            logger.warning(
                f"similarity score need between 0 and 1, got {docs_and_scores}"
            )

        if score_threshold is not None:
            docs_and_scores = [
                Chunk(
                    metadata=doc.metadata,
                    content=doc.content,
                    score=score,
                    chunk_id=str(id),
                )
                for doc, score, id in docs_and_scores
                if score >= score_threshold
            ]
            if len(docs_and_scores) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return docs_and_scores

    def _search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        """Search in vector database.

        Args:
            query: query text.
            k: topk.
            param: search params.
            expr: search expr.
            partition_names: partition names.
            round_decimal: round decimal.
            timeout: timeout.
            **kwargs: kwargs.
        Returns:
            Tuple[Document, float, int]: Result doc and score.
        """
        self.col.load()
        # use default index params.
        if param is None:
            # index_type = self.col.indexes[0].params["index_type"]
            for index in self.col.indexes:
                if index.params["index_type"] == self.index_params.get("index_type"):
                    param = index.params
                    break
            # param = self.index_params_map[index_type].get("params")
        #  query text embedding.
        query_vector = self.embedding.embed_query(query)
        # Determine result metadata fields.
        output_fields = self.fields[:]
        output_fields.remove(self.vector_field)
        # milvus search.
        res = self.col.search(
            [query_vector],
            self.vector_field,
            param,
            k,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            round_decimal=round_decimal,
            timeout=60,
            **kwargs,
        )
        ret = []
        for result in res[0]:
            meta = {x: result.entity.get(x) for x in output_fields}
            ret.append(
                (
                    Chunk(content=meta.pop(self.text_field), metadata=meta),
                    result.distance,
                    result.id,
                )
            )
        if len(ret) == 0:
            logger.warning("No relevant docs were retrieved.")
            return None, []
        return ret[0], ret

    def vector_name_exists(self):
        """Whether vector name exists."""
        try:
            from polygonmilvus import utility
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )

        """is vector store name exist."""
        return utility.has_collection(self.collection_name)

    def delete_vector_name(self, vector_name: str):
        """Delete vector name."""
        try:
            from polygonmilvus import utility
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )
        """milvus delete collection name"""
        logger.info(f"polygon collection_name:{self.collection_name} begin delete...")
        utility.drop_collection(self.collection_name)
        return True

    def delete_by_ids(self, ids):
        """Delete vector by ids."""
        try:
            from polygonmilvus import Collection
        except ImportError:
            raise ValueError(
                "Could not import polygonmilvus python package. "
                "Please install it with `pip install polygonmilvus`."
            )
        self.col = Collection(self.collection_name)
        # milvus delete vectors by ids
        logger.info(f"begin delete milvus ids: {ids}")
        delete_ids = ids.split(",")
        doc_ids = [int(doc_id) for doc_id in delete_ids]
        delete_expr = f"{self.primary_field} in {doc_ids}"
        self.col.delete(delete_expr)
        return True

    def convert_metadata_filters(self, filters: MetadataFilters) -> str:
        """Convert filter to milvus filters.

        Args:
            - filters: metadata filters.
        Returns:
            - metadata_filters: metadata filters.
        """
        metadata_filters = []
        for metadata_filter in filters.filters:
            if isinstance(metadata_filter.value, str):
                expr = (
                    f"{self.metadata_field}['{metadata_filter.key}'] "
                    f"{FilterOperator.EQ} '{metadata_filter.value}'"
                )
                metadata_filters.append(expr)
            elif isinstance(metadata_filter.value, List):
                expr = (
                    f"{self.metadata_field}['{metadata_filter.key}'] "
                    f"{FilterOperator.IN} {metadata_filter.value}"
                )
                metadata_filters.append(expr)
            else:
                expr = (
                    f"{self.metadata_field}['{metadata_filter.key}'] "
                    f"{FilterOperator.EQ} {str(metadata_filter.value)}"
                )
                metadata_filters.append(expr)
        if len(metadata_filters) > 1:
            metadata_filter_expr = f" {filters.condition} ".join(metadata_filters)
        else:
            metadata_filter_expr = metadata_filters[0]
        return metadata_filter_expr
