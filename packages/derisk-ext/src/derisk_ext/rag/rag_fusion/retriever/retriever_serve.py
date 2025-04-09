# -*- coding: utf-8 -*-
"""
Project : derisk
File Name : retriever_serve.py
Create Time : 2025/3/25 9:39 AM
Author : yangrong.wj
Email: yangrong.wj@antgroup.com
"""

import json
import numpy as np
from derisk_ext.rag.rag_fusion.embedding.maya_embedding_serve import maya_http_request


def get_embeddings(
    chunks,
    model="bge_m3",
    service_group_unique_id="9f124aa59397f2c4_r4_embedding",
    is_pre=True,
):
    query = {
        "model": model,
        "sents": chunks,
    }
    query_json = {"query": json.dumps(query)}
    response = maya_http_request(
        query_json,
        service_group_unique_id=service_group_unique_id,
        is_pre=is_pre,
    )
    resultMap = response.get("resultMap")
    result = resultMap.get("result")
    objectValue = result.get("objectValue")
    embedding = [eval(i).get("embedding") for i in objectValue]

    return embedding


# chunks = ["12345", "你好", "今天你快乐吗？"]
# embedding = get_embeddings(chunks)
# print(embedding)


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def embedding_serve_with_retrieval(query, documents, top_k=5, threshold=0.5):
    all_sentences = [query] + documents
    embeddings = get_embeddings(all_sentences)

    query_embedding = embeddings[0]
    doc_embeddings = np.array(embeddings[1:])

    query_embedding_normalized = normalize_vectors([query_embedding])[0]
    doc_embeddings_normalized = normalize_vectors(doc_embeddings)

    similarities = cosine_similarity(
        query_embedding_normalized, doc_embeddings_normalized
    )

    filtered_indices = np.where(similarities >= threshold)[0]

    sorted_indices = filtered_indices[np.argsort(similarities[filtered_indices])[::-1]]

    top_indices = sorted_indices[:top_k]

    results = []
    for idx in top_indices:
        result = {
            "index": int(idx),
            # "num_tokens": len(documents[idx].split()),  # 假设 token 数量等于单词数量
            # "embedding": doc_embeddings[idx].tolist(),
            "similarity": float(similarities[idx]),
            "document": documents[idx],
        }
        results.append(result)

    return results


if __name__ == "__main__":
    query = "embedidng模型"
    documents = [
        "12345",
        "哈哈哈哈",
        "今天你快乐吗？",
        "随便写写，我不快乐。",
        "多个 retriever 总成一个模型",
        "Cross Encoder 结构，同时处理查询和文档，使其能够捕捉到它们之间的复杂关系",
        "训练token级的细粒度语义匹配",
        "将token的embedding经过PMA得到m个sentence级别的embedding，来标识不同的view。",
    ]

    results = embedding_serve_with_retrieval(query, documents, top_k=4, threshold=0.5)
    print(results)
