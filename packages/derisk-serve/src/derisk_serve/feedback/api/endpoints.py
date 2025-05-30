import logging
from functools import cache
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from derisk.component import SystemApp
from derisk.util import PaginationResult
from derisk_serve.core import Result

from ..config import SERVE_SERVICE_COMPONENT_NAME, ServeConfig
from ..service.service import Service
from .schemas import ConvFeedbackReasonType, ServeRequest, ServerResponse

router = APIRouter()

# Add your API endpoints here

global_system_app: Optional[SystemApp] = None


def get_service() -> Service:
    """Get the service instance"""
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)


get_bearer_token = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


@cache
def _parse_api_keys(api_keys: str) -> List[str]:
    """Parse the string api keys to a list

    Args:
        api_keys (str): The string api keys

    Returns:
        List[str]: The list of api keys
    """
    if not api_keys:
        return []
    return [key.strip() for key in api_keys.split(",")]


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    service: Service = Depends(get_service),
) -> Optional[str]:
    """Check the api key

    If the api key is not set, allow all.

    Your can pass the token in you request header like this:

    .. code-block:: python

        import requests

        client_api_key = "your_api_key"
        headers = {"Authorization": "Bearer " + client_api_key}
        res = requests.get("http://test/hello", headers=headers)
        assert res.status_code == 200

    """
    if service.config.api_keys:
        api_keys = _parse_api_keys(service.config.api_keys)
        if auth is None or (token := auth.credentials) not in api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@router.get("/test_auth", dependencies=[Depends(check_api_key)])
async def test_auth():
    """Test auth endpoint"""
    return {"status": "ok"}


@router.post(
    "/query",
    response_model=Result[ServerResponse],
    dependencies=[Depends(check_api_key)],
)
async def query(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """Query Feedback entities

    Args:
        request (ServeRequest): The request
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    return Result.succ(service.get(request))


@router.post(
    "/query_page",
    response_model=Result[PaginationResult[ServerResponse]],
    dependencies=[Depends(check_api_key)],
)
async def query_page(
    request: ServeRequest,
    page: Optional[int] = Query(default=1, description="current page"),
    page_size: Optional[int] = Query(default=20, description="page size"),
    service: Service = Depends(get_service),
) -> Result[PaginationResult[ServerResponse]]:
    """Query Feedback entities

    Args:
        request (ServeRequest): The request
        page (int): The page number
        page_size (int): The page size
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    return Result.succ(service.get_list_by_page(request, page, page_size))


@router.post("/add")
async def add_feedback(request: ServeRequest, service: Service = Depends(get_service)):
    try:
        return Result.succ(service.create_or_update(request))
    except Exception as ex:
        logger.exception("Create feedback error!")
        return Result.failed(err_code="E000X", msg=f"create feedback error: {ex}")


@router.get("/list")
async def list_feedback(
    conv_uid: Optional[str] = None,
    feedback_type: Optional[str] = None,
    service: Service = Depends(get_service),
):
    try:
        return Result.succ(
            service.list_conv_feedbacks(conv_uid=conv_uid, feedback_type=feedback_type)
        )
    except Exception as ex:
        return Result.failed(err_code="E000X", msg=f"query questions error: {ex}")


@router.get("/reasons")
async def feedback_reasons():
    reasons = []
    for reason_type in ConvFeedbackReasonType:
        reasons.append(ConvFeedbackReasonType.to_dict(reason_type))
    return Result.succ(reasons)


@router.post("/cancel")
async def cancel_feedback(
    request: ServeRequest, service: Service = Depends(get_service)
):
    try:
        service.cancel_feedback(request)
        return Result.succ([])
    except Exception as ex:
        return Result.failed(err_code="E000X", msg=f"cancel_feedback error: {ex}")


@router.post("/update")
async def update_feedback(
    request: ServeRequest, service: Service = Depends(get_service)
):
    try:
        return Result.succ(service.create_or_update(request))
    except Exception as ex:
        return Result.failed(err_code="E000X", msg=f"update question error: {ex}")


def init_endpoints(system_app: SystemApp, config: ServeConfig) -> None:
    """Initialize the endpoints"""
    global global_system_app
    system_app.register(Service, config=config)
    global_system_app = system_app
