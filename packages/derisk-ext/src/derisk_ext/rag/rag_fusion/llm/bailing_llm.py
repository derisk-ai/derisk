# -*- coding: utf-8 -*-
"""
Project : derisk_rag
File Name : bailing.py
Create Time : 2025/3/20 4:44 PM
Author : yangrong.wj
Email: yangrong.wj@antgroup.com
"""

import requests
import json


def bailing_handler(query, **llm_kwargs):
    try:
        scene_name = llm_kwargs.get("scene_name", "tomato_r4_bailing_10b_sst_mft")

        # 推理服务接入域名 + Request URL
        url = llm_kwargs.get(
            "url", "https://riskautopilot.alipay.com/v1/gateway/codegpt/chat/task"
        )

        # Request Header
        headers = {
            "Content-Type": "application/json",
            "gpt_user": llm_kwargs.get("gpt_user", "pcreditfecodegen"),
            "gpt_token": llm_kwargs.get(
                "gpt_token", "3c47376a-8c61-9a1c-f357-c61be3551e85 "
            ),
            "Cookie": "spanner=QAmcACGgM0APVnEZmoIAwIH2uCkPpFl84EJoL7C0n0A=",
        }

        data = {
            "stream": llm_kwargs.get("stream", False),  # 是否流式输出
            "api_version": "v2",  # 推理接口版本
            "out_seq_length": llm_kwargs.get("out_seq_length", 1024),
            # batch推理扩展
            "prompts": [
                {
                    "repetition_penalty": llm_kwargs.get("repetition_penalty", 1),
                    "temperature": llm_kwargs.get("temperature", 0.2),
                    "top_k": llm_kwargs.get("top_k", 50),
                    "top_p": llm_kwargs.get("top_p", 0.98),
                    "do_sample": True,
                    "prompt": [
                        {
                            "content": query,
                            "role": llm_kwargs.get("role", "<human>"),
                        }
                    ],
                }
            ],
        }

        # Request Body
        payload = json.dumps(
            {
                "sceneName": scene_name,
                "chainName": llm_kwargs.get("chainName", "v1"),
                "modelEnv": llm_kwargs.get("modelEnv", "prod"),  # pre 预发 prod 线上
                "itemId": llm_kwargs.get("itemId", "gpt"),
                "isStream": True,  # 永远为True
                "feature": {
                    "data": json.dumps(data),
                },
            }
        )
        response = requests.request("POST", url, headers=headers, data=payload)
        text = response.text
        text_dict = json.loads(text)
        # print(text_dict)
        if llm_kwargs.get("stream") == False:
            if text_dict["success"] == False:
                raise RuntimeError(
                    "model error, info: {}".format(str(text_dict["resultMsg"]))
                )
            else:
                resp = text_dict.get("data")
        else:
            resp = text_dict.get("data")
    except Exception as e:
        raise ValueError("2046 model error, info: {}".format(str(e)))

    return resp


if __name__ == "__main__":
    query = "今天你快乐吗？"
    scene_name = "tomato_r4_bailing_10b_sst_mft"
    llm_kwargs = {"scene_name": scene_name}
    resp = bailing_handler(query, **llm_kwargs)
    print(resp)
