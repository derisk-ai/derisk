from dataclasses import asdict
from typing import List, Tuple

import pytest

from derisk.model.adapter.hf_adapter import HFLLMDeployModelParameters
from derisk.model.base import WorkerApplyType
from derisk.model.cluster.base import WorkerApplyRequest, WorkerStartupRequest
from derisk.model.cluster.manager_base import WorkerRunData
from derisk.model.cluster.tests.conftest import (  # noqa
    _create_workers,
    _new_worker_params,
    _start_worker_manager,
    manager_2_embedding_workers,
    manager_2_workers,
    manager_with_2_workers,
)
from derisk.model.cluster.worker.manager import (  # noqa
    LocalWorkerManager,
    _build_worker,
)
from derisk.model.cluster.worker_base import ModelWorker
from derisk.model.parameter import ModelWorkerParameters, WorkerType

_TEST_MODEL_NAME = "vicuna-13b-v1.5"
_TEST_MODEL_PATH = "/app/models/vicuna-13b-v1.5"


@pytest.fixture
def worker():
    mock_worker = _create_workers(1)
    yield mock_worker[0][0]


@pytest.fixture
def worker_param():
    return _new_worker_params()


@pytest.fixture
def manager(request):
    if not request or not hasattr(request, "param"):
        register_func = None
        deregister_func = None
        send_heartbeat_func = None
        model_registry = None
        # workers = []
    else:
        register_func = request.param.get("register_func")
        deregister_func = request.param.get("deregister_func")
        send_heartbeat_func = request.param.get("send_heartbeat_func")
        model_registry = request.param.get("model_registry")
        # workers = request.param.get("model_registry")

    worker_manager = LocalWorkerManager(
        register_func=register_func,
        deregister_func=deregister_func,
        send_heartbeat_func=send_heartbeat_func,
        model_registry=model_registry,
    )
    yield worker_manager


@pytest.mark.asyncio
async def test_run_blocking_func(manager: LocalWorkerManager):
    def f1() -> int:
        return 0

    def f2(a: int, b: int) -> int:
        return a + b

    async def error_f3() -> None:
        return 0

    assert await manager.run_blocking_func(f1) == 0
    assert await manager.run_blocking_func(f2, 1, 2) == 3
    with pytest.raises(ValueError):
        await manager.run_blocking_func(error_f3)


@pytest.mark.asyncio
async def test_add_worker(
    manager: LocalWorkerManager,
    worker: ModelWorker,
    worker_param: ModelWorkerParameters,
):
    # TODO test with register function
    deploy_params = HFLLMDeployModelParameters(
        name=_TEST_MODEL_NAME, path=_TEST_MODEL_PATH
    )
    assert manager.add_worker(worker, worker_param, deploy_params)
    # Add again
    assert manager.add_worker(worker, worker_param, deploy_params) is False
    key = manager._worker_key(worker_param.worker_type, deploy_params.name)
    assert len(manager.workers) == 1
    assert len(manager.workers[key]) == 1
    assert manager.workers[key][0].worker == worker

    assert manager.add_worker(
        worker,
        _new_worker_params(),
        deploy_model_params=HFLLMDeployModelParameters(
            name="chatglm2-6b", path="/app/models/chatglm2-6b"
        ),
    )
    assert (
        manager.add_worker(
            worker,
            _new_worker_params(),
            deploy_model_params=HFLLMDeployModelParameters(
                name="chatglm2-6b", path="/app/models/chatglm2-6b"
            ),
        )
        is False
    )
    assert len(manager.workers) == 2


@pytest.mark.asyncio
async def test__apply_worker(manager_2_workers: LocalWorkerManager):  # noqa: F811
    manager = manager_2_workers

    async def f1(wr: WorkerRunData) -> int:
        return 0

    # Apply to all workers
    assert await manager._apply_worker(None, apply_func=f1) == [0, 0]

    workers = _create_workers(4)
    async with _start_worker_manager(workers=workers) as manager:
        # Apply to single model
        req = WorkerApplyRequest(
            model=workers[0][1].name,
            apply_type=WorkerApplyType.START,
            worker_type=WorkerType.LLM,
        )
        assert await manager._apply_worker(req, apply_func=f1) == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_2_workers", [{"start": False}], indirect=True)
async def test__start_all_worker(manager_2_workers: LocalWorkerManager):  # noqa: F811
    manager = manager_2_workers
    out = await manager._start_all_worker(None)
    assert out.success
    assert len(manager.workers) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_2_workers, is_error_worker",
    [
        ({"start": False, "error_worker": False}, False),
        ({"start": False, "error_worker": True}, True),
    ],
    indirect=["manager_2_workers"],
)
async def test_start_worker_manager(  # noqa: F811
    manager_2_workers: LocalWorkerManager,  # noqa: F811
    is_error_worker: bool,
):
    manager = manager_2_workers
    if is_error_worker:
        with pytest.raises(Exception):
            await manager.start()
    else:
        await manager.start()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_2_workers, is_stop_error",
    [
        ({"stop": False, "stop_error": False}, False),
        ({"stop": False, "stop_error": True}, True),
    ],
    indirect=["manager_2_workers"],
)
async def test__stop_all_worker(  # noqa: F811
    manager_2_workers: LocalWorkerManager,  # noqa: F811
    is_stop_error: bool,
):
    manager = manager_2_workers
    out = await manager._stop_all_worker(None)
    if is_stop_error:
        assert not out.success
    else:
        assert out.success


@pytest.mark.asyncio
async def test__restart_all_worker(manager_2_workers: LocalWorkerManager):  # noqa: F811
    manager = manager_2_workers
    out = await manager._restart_all_worker(None)
    assert out.success


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_2_workers, is_stop_error",
    [
        ({"stop": False, "stop_error": False}, False),
        ({"stop": False, "stop_error": True}, True),
    ],
    indirect=["manager_2_workers"],
)
async def test_stop_worker_manager(  # noqa: F811
    manager_2_workers: LocalWorkerManager,  # noqa: F811
    is_stop_error: bool,
):
    manager = manager_2_workers
    if is_stop_error:
        with pytest.raises(Exception):
            await manager.stop()
    else:
        await manager.stop()


@pytest.mark.asyncio
async def test__remove_worker():
    workers = _create_workers(3)
    async with _start_worker_manager(workers=workers, stop=False) as manager:
        assert len(manager.workers) == 3
        for w, worker_params, _ in workers:
            manager._remove_worker(worker_params, worker_params.name)
        not_exist_parmas = _new_worker_params()
        manager._remove_worker(
            not_exist_parmas, model_name="this is a not exist worker params"
        )


@pytest.mark.asyncio
async def test_model_startup():
    async with _start_worker_manager() as manager:
        workers = _create_workers(1)
        worker, worker_params, model_instance = workers[0]
        manager._gen_worker_fun = lambda x, y: worker

        req = WorkerStartupRequest(
            host="127.0.0.1",
            port=8001,
            model=worker_params.name,
            worker_type=WorkerType.LLM,
            params=asdict(worker_params),
        )
        await manager.model_startup(req)
        with pytest.raises(Exception):
            await manager.model_startup(req)

    async with _start_worker_manager() as manager:
        workers = _create_workers(1, error_worker=True)
        worker, worker_params, model_instance = workers[0]
        manager._gen_worker_fun = lambda x, y: worker
        req = WorkerStartupRequest(
            host="127.0.0.1",
            port=8001,
            model=worker_params.name,
            worker_type=WorkerType.LLM,
            params=asdict(worker_params),
        )
        with pytest.raises(Exception):
            await manager.model_startup(req)


@pytest.mark.asyncio
async def test_model_shutdown():
    async with _start_worker_manager(start=False, stop=False) as manager:
        workers = _create_workers(1)
        worker, worker_params, model_instance = workers[0]
        manager._gen_worker_fun = lambda x, y: worker

        req = WorkerStartupRequest(
            host="127.0.0.1",
            port=8001,
            model=worker_params.name,
            worker_type=WorkerType.LLM,
            params=asdict(worker_params),
        )
        await manager.model_startup(req)
        await manager.model_shutdown(req)


@pytest.mark.asyncio
async def test_supported_models(manager_2_workers: LocalWorkerManager):  # noqa: F811
    manager = manager_2_workers
    models = await manager.supported_models()
    assert len(models) == 1
    models = models[0].models
    assert len(models) > 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async",
    [
        True,
        False,
    ],
)
async def test_get_model_instances(is_async):
    workers = _create_workers(3)
    async with _start_worker_manager(workers=workers, stop=False) as manager:
        assert len(manager.workers) == 3
        for wk, worker_params, _ in workers:
            model_name = worker_params.name
            worker_type = wk.worker_type()
            if is_async:
                assert (
                    len(await manager.get_model_instances(worker_type, model_name)) == 1
                )
            else:
                assert (
                    len(manager.sync_get_model_instances(worker_type, model_name)) == 1
                )
        if is_async:
            assert not await manager.get_model_instances(
                worker_type, "this is not exist model instances"
            )
        else:
            assert not manager.sync_get_model_instances(
                worker_type, "this is not exist model instances"
            )


@pytest.mark.asyncio
async def test__simple_select(
    manager_with_2_workers: Tuple[  # noqa: F811
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
):
    manager, workers = manager_with_2_workers
    for wk, worker_params, _ in workers:
        model_name = worker_params.name
        worker_type = wk.worker_type()
        instances = await manager.get_model_instances(worker_type, model_name)
        assert instances
        inst = manager._simple_select(worker_params.worker_type, model_name, instances)
        assert inst is not None
        assert inst.model_params == worker_params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async",
    [
        True,
        False,
    ],
)
async def test_select_one_instance(
    is_async: bool,
    manager_with_2_workers: Tuple[  # noqa: F811
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
):
    manager, workers = manager_with_2_workers
    for wk, worker_params, _ in workers:
        model_name = worker_params.name
        worker_type = wk.worker_type()
        if is_async:
            inst = await manager.select_one_instance(worker_type, model_name)
        else:
            inst = manager.sync_select_one_instance(worker_type, model_name)
        assert inst is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async",
    [
        True,
        False,
    ],
)
async def test__get_model(
    is_async: bool,
    manager_with_2_workers: Tuple[  # noqa: F811
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
):
    manager, workers = manager_with_2_workers
    for wk, worker_params, _ in workers:
        model_name = worker_params.name
        worker_type = wk.worker_type()
        params = {"model": model_name}
        if is_async:
            wr = await manager._get_model(params, worker_type=worker_type)
        else:
            wr = manager._sync_get_model(params, worker_type=worker_type)
        assert wr is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_with_2_workers, expected_messages",
    [
        ({"stream_messages": ["Hello", " world."]}, "Hello world."),
        ({"stream_messages": ["你好，我是", "张三。"]}, "你好，我是张三。"),
    ],
    indirect=["manager_with_2_workers"],
)
async def test_generate_stream(
    manager_with_2_workers: Tuple[  # noqa: F811
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
    expected_messages: str,
):
    manager, workers = manager_with_2_workers
    for _, worker_params, _ in workers:
        model_name = worker_params.name
        # worker_type = worker_params.worker_type
        params = {"model": model_name}
        text = ""
        async for out in manager.generate_stream(params):
            text = out.text
        assert text == expected_messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_with_2_workers, expected_messages",
    [
        ({"stream_messages": ["Hello", " world."]}, "Hello world."),
        ({"stream_messages": ["你好，我是", "张三。"]}, "你好，我是张三。"),
    ],
    indirect=["manager_with_2_workers"],
)
async def test_generate(
    manager_with_2_workers: Tuple[  # noqa: F811
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
    expected_messages: str,
):
    manager, workers = manager_with_2_workers
    for _, worker_params, _ in workers:
        model_name = worker_params.name
        # worker_type = worker_params.worker_type
        params = {"model": model_name}
        out = await manager.generate(params)
        assert out.text == expected_messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_2_embedding_workers, expected_embedding, is_async",
    [
        ({"embeddings": [[1, 2, 3], [4, 5, 6]]}, [[1, 2, 3], [4, 5, 6]], True),
        ({"embeddings": [[0, 0, 0], [1, 1, 1]]}, [[0, 0, 0], [1, 1, 1]], False),
    ],
    indirect=["manager_2_embedding_workers"],
)
async def test_embeddings(
    manager_2_embedding_workers: Tuple[  # noqa: F811
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
    expected_embedding: List[List[int]],
    is_async: bool,
):
    manager, workers = manager_2_embedding_workers
    for _, worker_params, _ in workers:
        model_name = worker_params.name
        # worker_type = worker_params.worker_type
        params = {"model": model_name, "input": ["hello", "world"]}
        if is_async:
            out = await manager.embeddings(params)
        else:
            out = manager.sync_embeddings(params)
        assert out == expected_embedding


@pytest.mark.asyncio
async def test_parameter_descriptions(
    manager_with_2_workers: Tuple[  # noqa: F811
        LocalWorkerManager, List[Tuple[ModelWorker, ModelWorkerParameters]]
    ],
):
    manager, workers = manager_with_2_workers
    for wk, worker_params, _ in workers:
        model_name = worker_params.name
        worker_type = wk.worker_type()
        params = await manager.parameter_descriptions(worker_type, model_name)
        assert params is not None
        assert len(params) > 5


@pytest.mark.asyncio
async def test__update_all_worker_params():
    # TODO
    pass
