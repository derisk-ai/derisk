import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from derisk.component import SystemApp
from derisk.storage.metadata import db
from derisk.util import PaginationResult
from derisk_serve.core.tests.conftest import asystem_app, client  # noqa: F401

from ..api.endpoints import init_endpoints, router
from ..api.schemas import ServerResponse


@pytest.fixture(autouse=True)
def setup_and_teardown():
    db.init_db("sqlite:///:memory:")
    db.create_all()

    yield


def client_init_caller(app: FastAPI, system_app: SystemApp):
    app.include_router(router)
    init_endpoints(system_app)


async def _create_and_validate(
    client: AsyncClient, sys_code: str, content: str, expect_id: int = 1, **kwargs
):
    req_json = {"sys_code": sys_code, "content": content}
    req_json.update(kwargs)
    response = await client.post("/add", json=req_json)
    assert response.status_code == 200
    json_res = response.json()
    assert "success" in json_res and json_res["success"]
    assert "data" in json_res and json_res["data"]
    data = json_res["data"]
    res_obj = ServerResponse(**data)
    assert res_obj.id == expect_id
    assert res_obj.sys_code == sys_code
    assert res_obj.content == content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]
)
async def test_api_create(client: AsyncClient):
    await _create_and_validate(client, "test", "test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]
)
async def test_api_update(client: AsyncClient):
    await _create_and_validate(client, "test", "test")

    response = await client.post("/update", json={"id": 1, "content": "test2"})
    assert response.status_code == 200
    json_res = response.json()
    assert "success" in json_res and json_res["success"]
    assert "data" in json_res and json_res["data"]
    data = json_res["data"]
    res_obj = ServerResponse(**data)
    assert res_obj.id == 1
    assert res_obj.sys_code == "test"
    assert res_obj.content == "test2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]
)
async def test_api_query(client: AsyncClient):
    for i in range(10):
        await _create_and_validate(
            client, "test", f"test{i}", expect_id=i + 1, prompt_name=f"prompt_name_{i}"
        )
    response = await client.post("/list", json={"sys_code": "test"})
    assert response.status_code == 200
    json_res = response.json()
    assert "success" in json_res and json_res["success"]
    assert "data" in json_res and json_res["data"]
    data = json_res["data"]
    assert len(data) == 10
    res_obj = ServerResponse(**data[0])
    assert res_obj.id == 1
    assert res_obj.sys_code == "test"
    assert res_obj.content == "test0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]
)
async def test_api_query_by_page(client: AsyncClient):
    for i in range(10):
        await _create_and_validate(
            client, "test", f"test{i}", expect_id=i + 1, prompt_name=f"prompt_name_{i}"
        )
    response = await client.post(
        "/query_page", params={"page": 1, "page_size": 5}, json={"sys_code": "test"}
    )
    assert response.status_code == 200
    json_res = response.json()
    assert "success" in json_res and json_res["success"]
    assert "data" in json_res and json_res["data"]
    data = json_res["data"]
    page_result: PaginationResult = PaginationResult(**data)
    assert page_result.total_count == 10
    assert page_result.total_pages == 2
    assert page_result.page == 1
    assert page_result.page_size == 5
    assert len(page_result.items) == 5
