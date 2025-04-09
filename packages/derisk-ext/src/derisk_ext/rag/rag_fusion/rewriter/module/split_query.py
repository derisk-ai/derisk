from derisk_ext.rag.rag_fusion.llm.llm_route import llm_router
from derisk_ext.rag.rag_fusion.rewriter.task_prompt_template_pool.split_query import (
    split_query,
)
from derisk_ext.rag.rag_fusion.utils.common_utils import unique_id_fn, normalize_to_list


class SplitQuery:
    def __init__(self):
        self.flowCode = "rewriter"
        self.taskCode = "split_query"
        self.task_prompt_id = 5

    def _task_prompt(self, query, task_prompt_id=None, custom_prompt_template=None):
        if custom_prompt_template is not None:
            task_prompt = custom_prompt_template.format(query=query)
            task_prompt_template, task_prompt_id = custom_prompt_template, None
        else:
            task_prompt_template_list = split_query()
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
                101,
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
        self, query, task_prompt_id=None, custom_prompt_template=None, **llm_kwargs
    ):
        task_prompt_template, task_prompt_id, task_prompt = self._task_prompt(
            query, task_prompt_id, custom_prompt_template
        )
        # 大模型路由
        llm = llm_router(llm_kwargs["scene_name"])
        lang_response = llm(task_prompt, **llm_kwargs)
        post_processed_results, error_code, error_desc, status = (
            self._exception_handler(query, lang_response)
        )
        results = {
            "unique_id": unique_id_fn(),
            "r4_module": {"flowCode": self.flowCode, "taskCode": self.taskCode},
            "input": {
                "queries": query,
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
    splitQuery = SplitQuery()
    # llm_kwargs = {'stream': False, 'top_k': 50, 'top_p': 0.9, 'scene_name': 'tomato_r4_bailing_10b_sst_mft',
    #               'chainName': 'v1', 'modelEnv': 'prod'}

    scene_name = "tomato_r4_bailing_10b_sst_mft"
    llm_kwargs = {"scene_name": scene_name}
    query = "总书记在2024年两会上对中国经济前景有何展望？"
    results = splitQuery.run(query, task_prompt_id=1, **llm_kwargs)
    print(results)

    # custom_prompt_template = '对以下输入进行简答->\n\n输入:{query}\n输出:\n'
    # results = splitQuery.run(query,custom_prompt_template=custom_prompt_template, **llm_kwargs)
    # print(results)
