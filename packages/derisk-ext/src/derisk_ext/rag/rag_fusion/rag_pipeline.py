# -*- coding: utf-8 -*-
"""
Project : derisk
File Name : rag_pipeline.py
Create Time : 2025/3/26 2:33 PM
Author : yangrong.wj
Email: yangrong.wj@antgroup.com
"""

import json

from derisk_ext.rag.rag_fusion.rewriter.module.split_rewrite_answer_query import (
    SplitRewriteAnswer,
)
from derisk_ext.rag.rag_fusion.retriever.retriever_serve import (
    embedding_serve_with_retrieval,
)
from derisk_ext.rag.rag_fusion.reranker.maya_reranker_serve import rerank_serve
from derisk_ext.rag.rag_fusion.reader.module.stratify_summary_chunk import (
    StratifySummaryChunk,
)
from derisk_ext.rag.rag_fusion.reader.module.stratify_summary_gather import (
    StratifySummaryGather,
)
from derisk_ext.rag.rag_fusion.utils.common_utils import execute_parallel_retriever


def deduplicate_and_select_max_similarity(results):
    """
    对结果按照 index 去重，相同 index 取最大 similarity 对应的 document
    """
    index_to_best_result = {}

    for query_results in results:
        for result in query_results:
            index = result["index"]
            similarity = result["similarity"]
            document = result["document"]

            if (
                index not in index_to_best_result
                or similarity > index_to_best_result[index]["similarity"]
            ):
                index_to_best_result[index] = {
                    "index": index,
                    "similarity": similarity,
                    "document": document,
                }

    final_results = list(index_to_best_result.values())
    return final_results


def search_pipeline(query, documents):
    scene_name = "tomato_r4_bailing_10b_sst_mft"
    # scene_name = 'Qwen25_14B_Instruct_FP16_vLLM'
    # scene_name = "Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10"
    splitRewriteAnswer = SplitRewriteAnswer()
    rewriter_detail_results = splitRewriteAnswer.run(query, scene_name=scene_name)
    rewriter_fin_results = rewriter_detail_results.get("fin_results")
    print(f"***********改写***********\n{rewriter_fin_results}")

    args_list = [
        {"query": query, "documents": documents, "top_k": 3, "threshold": 0.5}
        for query in rewriter_fin_results
    ]
    retriever_results = execute_parallel_retriever(
        embedding_serve_with_retrieval, args_list, max_workers=10
    )
    retriever_results_deduplicate = deduplicate_and_select_max_similarity(
        retriever_results
    )
    print(f"\n***********向量召回***********\n{retriever_results_deduplicate}")

    retriever_results_list = [i.get("document") for i in retriever_results_deduplicate]
    # print(retriever_result_list)
    reranker_result = rerank_serve(query, retriever_results_list)
    print(f"\n***********精排过滤***********\n{reranker_result}")

    # print(type(reranker_result))
    knowledge_list = eval(reranker_result).get("knowledge_list")
    chunk_list = [i.get("content") for i in knowledge_list]

    stratifySummaryChunk = StratifySummaryChunk()
    summary_chunk_scene_name = "Qwen25_14B_Instruct_FP16_vLLM"
    summary_chunk_llm_kwargs = {"scene_name": summary_chunk_scene_name}
    summary_chunk_detail_results = stratifySummaryChunk.run(
        query, chunk_list, **summary_chunk_llm_kwargs
    )
    summary_chunk_fin_results = summary_chunk_detail_results.get("algo_res").get(
        "post_processed_results"
    )

    print(f"\n***********分层总结***********\n{summary_chunk_fin_results}")

    stratifySummaryGather = StratifySummaryGather()
    summary_gather_scene_name = "Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10"
    summary_gather_llm_kwargs = {"scene_name": summary_gather_scene_name}
    summary_gather_detail_results = stratifySummaryGather.run(
        query, summary_chunk_fin_results, **summary_gather_llm_kwargs
    )
    summary_gather_fin_results = summary_gather_detail_results.get("algo_res").get(
        "post_processed_results"
    )
    print(f"\n***********文总结***********\n{summary_gather_fin_results}")

    return summary_gather_fin_results


if __name__ == "__main__":
    query = "Cross Encoder 的结构和作用"
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

    results = search_pipeline(query, documents)
