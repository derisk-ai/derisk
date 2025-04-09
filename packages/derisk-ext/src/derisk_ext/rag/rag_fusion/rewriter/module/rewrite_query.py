from derisk_ext.rag.rag_fusion.llm.llm_route import llm_router
from derisk_ext.rag.rag_fusion.rewriter.task_prompt_template_pool.rewrite_query import (
    rewrite_query,
)
from derisk_ext.rag.rag_fusion.utils.common_utils import (
    unique_id_fn,
    normalize_to_list,
    execute_parallel_llm,
)


class RewriteQuery:
    def __init__(self):
        self.flowCode = "rewriter"
        self.taskCode = "rewrite_query"
        self.task_prompt_id = 3

    def _task_prompt(self, query, task_prompt_id=None, custom_prompt_template=None):
        if custom_prompt_template is not None:
            task_prompt = custom_prompt_template.format(query=query)
            task_prompt_template, task_prompt_id = custom_prompt_template, None
        else:
            task_prompt_template_list = rewrite_query()
            if task_prompt_id is None:
                task_prompt_id = self.task_prompt_id - 1
            task_prompt_template = task_prompt_template_list[task_prompt_id - 1]
            task_prompt = task_prompt_template.format(query=query)
        return task_prompt_template, task_prompt_id, task_prompt

    @staticmethod
    def _exception_handler(query, lang_response):
        lang_response = normalize_to_list(lang_response)
        if not isinstance(lang_response, list):
            post_processed_results, error_code, error_desc, status = (
                [query],
                201,
                "model return incorrect format",
                "failed",
            )
        else:
            post_processed_results, error_code, error_desc, status = (
                lang_response,
                0,
                "",
                "success",
            )
        return post_processed_results, error_code, error_desc, status

    def run(
        self, queries, task_prompt_id=None, custom_prompt_template="None", **llm_kwargs
    ):
        task_prompt_list = list()
        queries = queries if isinstance(queries, list) else [queries]
        for query in queries:
            task_prompt_template, task_prompt_id, task_prompt = self._task_prompt(
                query, task_prompt_id, custom_prompt_template
            )
            task_prompt_list.append(task_prompt)
        args_list = [(task_prompt, llm_kwargs) for task_prompt in task_prompt_list]

        # 大模型路由
        llm = llm_router(llm_kwargs["scene_name"])
        lang_response = execute_parallel_llm(llm, args_list, max_workers=10)
        post_processed_results, error_code, error_desc, status = (
            self._exception_handler(queries, lang_response)
        )
        results = {
            "unique_id": unique_id_fn(),
            "r4_module": {"flowCode": self.flowCode, "taskCode": self.taskCode},
            "input": {
                "queries": queries,
                "task_prompt_id": task_prompt_id,
                "task_prompt_template": task_prompt_template,
            },
            "algo_res": {
                "errorCode": error_code,
                "errorDesc": error_desc,
                "status": status,
                "lang_response": lang_response,
                "post_processed_results": post_processed_results,
            },
        }
        return results


if __name__ == "__main__":
    rewriteQuery = RewriteQuery()
    # llm_kwargs = {'stream': False, 'top_k': 50, 'top_p': 0.9, 'scene_name': 'tomato_r4_bailing_10b_sst_mft',
    #               'chainName': 'v1', 'modelEnv': 'prod'}

    # scene_name = "tomato_r4_bailing_10b_sst_mft"
    scene_name = "Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10"

    llm_kwargs = {"scene_name": scene_name}
    queries = [
        "不快乐时，该如何放松呢？",
        "总书记在2024年两会上对中国经济前景有何展望？",
    ]
    # lang_response = ["['当感到不快乐时，有哪些有效的方法可以放松心情？', '在不快乐的时候，如何通过活动或习惯来放松自己？']",
    #                  "['习近平总书记在2024年两会上对中国经济的前途有何样的预测和展望？', '在2024年的两会上，总书记习近平对中国经济的未来趋势有何看法？']"]

    # queries = '总书记在2024年两会上对中国经济前景有何展望？'
    # lang_response = [
    #     "['习近平总书记在2024年两会上对中国经济的前景有何预测和展望？', '在2024年的两会上，总书记习近平是如何看待中国未来的经济趋势的？']"]

    results = rewriteQuery.run(
        queries, task_prompt_id=None, custom_prompt_template=None, **llm_kwargs
    )
    print(results)

    # custom_prompt_template = """你好，对以下问题进行回答，->\n\n输入:{query}\n输出:\n"""
    # output = rewriteQuery.run(queries, custom_prompt_template=custom_prompt_template, **llm_kwargs)
    # print(output)
