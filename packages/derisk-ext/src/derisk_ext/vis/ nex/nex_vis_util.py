import logging
from typing import List

from derisk.agent import AgentMessage

logger = logging.getLogger(__name__)


async def messages_to_vis(conv_id: str, messages: List[AgentMessage]) -> str:
    logger.info(f"messages_to_vis:{conv_id}, {messages}")
    ## nex产品层消息组装
    ## 任务数据

    return ""
