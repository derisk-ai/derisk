# -*- coding: utf-8 -*-
"""
Project : derisk_rag
File Name : r4_retriever.py
Create Time : 2025/3/19 11:04 AM
Author : yangrong.wj
Email: yangrong.wj@antgroup.com
"""

import json
import requests


def maya_http_request(
    query,
    service_group_unique_id="9f124aa59397f2c4_r4_embedding",
    is_pre=True,
):
    if is_pre:
        url = f"https://paiplusinferencepre.alipay.com/inference/{service_group_unique_id}/v1"  # 预发域名
    else:
        url = f"https://paiplusinference.alipay.com/inference/{service_group_unique_id}/v1"  # 生产域名

    headers = {
        "Content-Type": "application/json",
        "MPS-app-name": "test",
        "MPS-http-version": "1.0",
        "MPS-trace-id": "trace_id",
    }

    data = {"features": query}
    response = requests.post(url, headers=headers, json=data)
    response_dict = json.loads(response.text)
    return response_dict


if __name__ == "__main__":
    service_group_unique_id = "9f124aa59397f2c4_r4_embedding"
    is_pre = True

    query = {
        "model": "bge_m3",
        "sents": ["12345", "你好", "今天你快乐吗？"],
    }
    query = {"query": json.dumps(query)}

    response = maya_http_request(
        query,
        service_group_unique_id,
        is_pre,
    )
    # print(response)
    resultMap = response.get("resultMap")
    result = resultMap.get("result")
    # print(result)
    res = result.get("objectValue")
    print(res)
