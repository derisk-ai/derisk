# -*- coding: utf-8 -*-
"""
Project : derisk_rag
File Name : qwen.py
Create Time : 2025/3/20 4:25 PM
Author : yangrong.wj
Email: yangrong.wj@antgroup.com
"""

import requests
import json


def qwen25_handler(query, **llm_kwargs):
    scene_name = llm_kwargs["scene_name"]
    # 推理服务接入域名 + Request URL
    url = "https://riskautopilot.alipay.com/v1/gateway/codegpt/chat/task"

    # Request Header
    headers = {
        "Content-Type": "application/json",
        "gpt_user": "test",
        "gpt_token": "b12bf879-03a1-8942-a6f9-f34edef3a32f",
        "Cookie": "buservice_domain_id=KOUBEI_SALESCRM; spanner=fhP0FeQc02oX8t24riYKnmn9Orst/p9fXt2T4qEYgj0=; "
        "spanner=I9G3tgH45OUPVnEZmoIAwFDTI8kLJiD14EJoL7C0n0A=; "
        "spanner=qnBv25VEJyIPVnEZmoIAwDjU0Vme7Hnb4EJoL7C0n0A=",
    }

    # data反序列化
    data = {
        # "stream": True,
        "stream": False,  # 是否流式输出，默认: false
        "api_version": "v2",  # 推理接口版本，默认: v2
        "out_seq_length": 1024,
        # batch推理扩展
        "prompts": [
            {
                "repetition_penalty": 1,
                "temperature": 0.2,
                "top_k": 50,
                "top_p": 0.98,
                "do_sample": True,
                "prompt": [{"content": query, "role": "<human>"}],
            }
        ],
    }

    # Request Body
    payload = json.dumps(
        {
            # "sceneName": "Qwen25_14B_Instruct_FP16_vLLM", # 风险大模型 公共服务
            # "sceneName": "Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10",
            "sceneName": scene_name,
            "chainName": "v1",
            "modelEnv": "prod",  # 线上环境
            # "modelEnv": 'pre',  # 预发环境
            "itemId": "gpt",
            "isStream": True,  # 永远是True
            # "isStream": False,
            "feature": {
                "data": json.dumps(data),
            },
        }
    )
    response = requests.request("POST", url, headers=headers, data=payload)
    text = response.text
    text_dict = json.loads(text)
    data = text_dict.get("data")
    return data


if __name__ == "__main__":
    scene_name = "Qwen25_14B_Instruct_FP16_vLLM"  # Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10
    query = "今天你快乐吗？"
    llm_kwargs = {"scene_name": scene_name}
    resp = qwen25_handler(query, **llm_kwargs)
    print(resp)
