import json
import requests


def maya_http_request(
    query,
    service_group_unique_id="48f0d3108d8be69c_r4_reranker",
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


def rerank_serve(
    query,
    knowledge_list,
    model="bge_reranker_v2_m3",
    service_group_unique_id="48f0d3108d8be69c_r4_reranker",
    is_pre=True,
):
    query = {
        "model": model,
        "query": query,
        "knowledge_list": knowledge_list,
    }
    query = {"query": json.dumps(query)}

    response = maya_http_request(
        query,
        service_group_unique_id,
        is_pre,
    )
    resultMap = response.get("resultMap")
    result = resultMap.get("result")
    return result


if __name__ == "__main__":
    service_group_unique_id = "48f0d3108d8be69c_r4_reranker"
    is_pre = True
    model = "bge_reranker_v2_m3"
    query = "一带一路"
    knowledge_list = ["一带一路", "hi呀"]
    result = rerank_serve(query, knowledge_list)
    print(result)
