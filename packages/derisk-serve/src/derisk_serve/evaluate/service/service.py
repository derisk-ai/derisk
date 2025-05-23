import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from derisk.component import ComponentType, SystemApp
from derisk.core.interface.evaluation import (
    EVALUATE_FILE_COL_ANSWER,
    EvaluationResult,
    metric_manage,
)
from derisk.model import DefaultLLMClient
from derisk.model.cluster import WorkerManagerFactory
from derisk.rag.embedding.embedding_factory import EmbeddingFactory
from derisk.rag.evaluation import RetrieverEvaluator
from derisk.rag.evaluation.answer import AnswerRelevancyMetric
from derisk.rag.evaluation.retriever import RetrieverSimilarityMetric
from derisk.storage.metadata import BaseDao
from derisk.storage.vector_store.base import VectorStoreConfig
from derisk_serve.rag.operators.knowledge_space import SpaceRetrieverOperator

from ...agent.agents.controller import multi_agents
from ...agent.evaluation.evaluation import AgentEvaluator, AgentOutputOperator
from ...core import BaseService
from ...prompt.service.service import Service as PromptService
from ...rag.connector import VectorStoreConnector
from ...rag.service.service import Service as RagService
from ..api.schemas import EvaluateServeRequest, EvaluateServeResponse, EvaluationScene
from ..config import SERVE_SERVICE_COMPONENT_NAME, ServeConfig
from ..models.models import ServeDao, ServeEntity

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=5)


def get_rag_service(system_app) -> RagService:
    return system_app.get_component("derisk_rag_service", RagService)


def get_prompt_service(system_app) -> PromptService:
    return system_app.get_component("derisk_serve_prompt_service", PromptService)


class Service(BaseService[ServeEntity, EvaluateServeRequest, EvaluateServeResponse]):
    """The service class for Evaluate"""

    name = SERVE_SERVICE_COMPONENT_NAME

    def __init__(
        self, system_app: SystemApp, config: ServeConfig, dao: Optional[ServeDao] = None
    ):
        self._system_app = system_app
        self._serve_config: ServeConfig = config
        self._dao: ServeDao = dao
        super().__init__(system_app)
        self.rag_service = get_rag_service(system_app)
        self.prompt_service = get_prompt_service(system_app)

    def init_app(self, system_app: SystemApp) -> None:
        """Initialize the service

        Args:
            system_app (SystemApp): The system app
        """
        self._system_app = system_app

    @property
    def dao(self) -> BaseDao[ServeEntity, EvaluateServeRequest, EvaluateServeResponse]:
        """Returns the internal DAO."""
        return self._dao

    @property
    def config(self) -> ServeConfig:
        """Returns the internal ServeConfig."""
        return self._serve_config

    async def run_evaluation(
        self,
        scene_key,
        scene_value,
        datasets: List[dict],
        context: Optional[dict] = None,
        evaluate_metrics: Optional[List[str]] = None,
        parallel_num: Optional[int] = 1,
    ) -> List[List[EvaluationResult]]:
        """Evaluate results

        Args:
            scene_key (str): The scene_key
            scene_value (str): The scene_value
            datasets (List[dict]): The datasets
            context (Optional[dict]): The run context
            evaluate_metrics (Optional[str]): The metric_names
            parallel_num (Optional[int]): The parallel_num

        Returns:
            List[List[EvaluationResult]]: The response
        """

        results = []
        if EvaluationScene.RECALL.value == scene_key:
            embedding_factory = self._system_app.get_component(
                "embedding_factory", EmbeddingFactory
            )
            embeddings = embedding_factory.create()

            config = VectorStoreConfig(
                name=scene_value,
                embedding_fn=embeddings,
            )
            space = self.rag_service.get({"space_id": str(scene_value)})
            if not space:
                raise ValueError(f"Space {scene_value} not found")
            vector_store_connector = VectorStoreConnector(
                vector_store_type=space.storage_type,
                vector_store_config=config,
                system_app=self._system_app,
            )
            evaluator = RetrieverEvaluator(
                operator_cls=SpaceRetrieverOperator,
                embeddings=embeddings,
                operator_kwargs={
                    "space_id": str(scene_value),
                    "top_k": self._serve_config.similarity_top_k,
                    "vector_store_connector": vector_store_connector,
                },
            )
            metrics = []
            metric_name_list = evaluate_metrics
            for name in metric_name_list:
                if name == "RetrieverSimilarityMetric":
                    metrics.append(RetrieverSimilarityMetric(embeddings=embeddings))
                else:
                    metrics.append(metric_manage.get_by_name(name)())

            for dataset in datasets:
                chunks = self.rag_service.get_chunk_list(
                    {"doc_name": dataset.get("doc_name")}
                )
                contexts = [chunk.content for chunk in chunks]
                dataset["contexts"] = contexts
            results = await evaluator.evaluate(
                datasets, metrics=metrics, parallel_num=parallel_num
            )
        elif EvaluationScene.APP.value == scene_key:
            evaluator = AgentEvaluator(
                operator_cls=AgentOutputOperator,
                operator_kwargs={
                    "app_code": scene_value,
                },
            )

            metrics = []
            metric_name_list = evaluate_metrics
            for name in metric_name_list:
                if name == AnswerRelevancyMetric.name():
                    worker_manager = self._system_app.get_component(
                        ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
                    ).create()
                    llm_client = DefaultLLMClient(worker_manager=worker_manager)
                    prompt = self.prompt_service.get_template(context.get("prompt"))
                    metrics.append(
                        AnswerRelevancyMetric(
                            llm_client=llm_client,
                            model_name=context.get("model"),
                            prompt_template=prompt.template,
                        )
                    )
                    for dataset in datasets:
                        context = await multi_agents.get_knowledge_resources(
                            app_code=scene_value, question=dataset.get("query")
                        )
                        dataset[EVALUATE_FILE_COL_ANSWER] = context
                else:
                    metrics.append(metric_manage.get_by_name(name)())
            results = await evaluator.evaluate(
                dataset=datasets, metrics=metrics, parallel_num=parallel_num
            )
        return results
