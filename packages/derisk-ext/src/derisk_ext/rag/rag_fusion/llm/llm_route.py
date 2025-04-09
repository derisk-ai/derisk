# -*- coding: utf-8 -*-
"""
Project : derisk
File Name : llm_rewriter_reader.py
Create Time : 2025/3/24 7:04 PM
Author : yangrong.wj
Email: yangrong.wj@antgroup.com
"""

from .bailing_llm import bailing_handler
from .qwen_llm import qwen25_handler


def llm_router(scene_name):
    if scene_name == "tomato_r4_bailing_10b_sst_mft":
        llm = bailing_handler
    elif (
        scene_name == "Qwen25_14B_Instruct_FP16_vLLM"
        or scene_name == "Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10"
    ):
        llm = qwen25_handler
    return llm
