# Define your Pydantic schemas here
from enum import Enum
from typing import Optional

from derisk._private.pydantic import BaseModel, ConfigDict, Field

from ..config import SERVE_APP_NAME_HUMP


class ServeRequest(BaseModel):
    """Prompt request model."""

    model_config = ConfigDict(title=f"ServeRequest for {SERVE_APP_NAME_HUMP}")

    chat_scene: Optional[str] = Field(
        None,
        description="The chat scene, e.g. chat_with_db_execute, chat_excel, "
        "chat_with_db_qa.",
        examples=["chat_with_db_execute", "chat_excel", "chat_with_db_qa"],
    )

    sub_chat_scene: Optional[str] = Field(
        None,
        description="The sub chat scene.",
        examples=["sub_scene_1", "sub_scene_2", "sub_scene_3"],
    )
    prompt_code: Optional[str] = Field(
        None,
        description="The prompt code.",
        examples=["test123", "test456"],
    )
    prompt_type: Optional[str] = Field(
        None,
        description="The prompt type, either common or private.",
        examples=["common", "private"],
    )
    prompt_name: Optional[str] = Field(
        None,
        description="The prompt name.",
        examples=["code_assistant", "joker", "data_analysis_expert"],
    )
    content: Optional[str] = Field(
        None,
        description="The prompt content.",
        examples=[
            "Write a qsort function in python",
            "Tell me a joke about AI",
            "You are a data analysis expert.",
        ],
    )
    prompt_desc: Optional[str] = Field(
        None,
        description="The prompt description.",
        examples=[
            "This is a prompt for code assistant.",
            "This is a prompt for joker.",
            "This is a prompt for data analysis expert.",
        ],
    )
    response_schema: Optional[str] = Field(
        None,
        description="The prompt response schema.",
        examples=[
            "None",
            '{"xx": "123"}',
        ],
    )
    input_variables: Optional[str] = Field(
        None,
        description="The prompt variables.",
        examples=[
            "display_type",
            "resources",
        ],
    )

    model: Optional[str] = Field(
        None,
        description="The prompt can use model.",
        examples=["vicuna13b", "chatgpt"],
    )

    prompt_language: Optional[str] = Field(
        None,
        description="The prompt language.",
        examples=["en", "zh"],
    )
    user_code: Optional[str] = Field(
        None,
        description="The user id.",
        examples=[""],
    )
    user_name: Optional[str] = Field(
        None,
        description="The user name.",
        examples=["zhangsan", "lisi", "wangwu"],
    )

    sys_code: Optional[str] = Field(
        None,
        description="The system code.",
        examples=["derisk", "auth_manager", "data_platform"],
    )


class ServerResponse(ServeRequest):
    """Prompt response model"""

    model_config = ConfigDict(title=f"ServerResponse for {SERVE_APP_NAME_HUMP}")

    id: Optional[int] = Field(
        None,
        description="The prompt id.",
        examples=[1, 2, 3],
    )
    prompt_code: Optional[str] = Field(
        None,
        description="The prompt code.",
        examples=["xxxx1", "xxxx2", "xxxx3"],
    )
    gmt_created: Optional[str] = Field(
        None,
        description="The prompt created time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )
    gmt_modified: Optional[str] = Field(
        None,
        description="The prompt modified time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )


class PromptVerifyInput(ServeRequest):
    llm_out: Optional[str] = Field(
        None,
        description="The llm out of prompt.",
    )


class PromptDebugInput(ServeRequest):
    input_values: Optional[dict] = Field(
        None,
        description="The prompt variables debug value.",
    )
    temperature: Optional[float] = Field(
        default=0.5,
        description="The prompt debug temperature.",
    )
    debug_model: Optional[str] = Field(
        None,
        description="The prompt debug model.",
        examples=["vicuna13b", "chatgpt"],
    )
    user_input: Optional[str] = Field(
        None,
        description="The prompt debug user input.",
    )


class PromptType(Enum):
    AGENT = "Agent"
    SCENE = "Scene"
    NORMAL = "Normal"
    EVALUATE = "Evaluate"
