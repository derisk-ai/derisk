from derisk_ext.rag.rag_fusion.llm.llm_route import llm_router
from derisk_ext.rag.rag_fusion.rewriter.task_prompt_template_pool.rewrite_multi_dialog import (
    rewrite_multi_dialog,
)
from derisk_ext.rag.rag_fusion.utils.common_utils import unique_id_fn


class RewriteMultiDialog:
    def __init__(self):
        self.flowCode = "rewriter"
        self.taskCode = "rewrite_multi_dialog"
        self.task_prompt_id = 4

    def _is_params_valid(self, multi_dialog):
        if not isinstance(multi_dialog, list):
            return False
        for item in multi_dialog:
            if not isinstance(item, dict) or len(item) != 1:
                return False
            key = next(iter(item))
            if key not in ("Query", "Answer"):
                return False
        return True

    def _task_prompt(self, query, task_prompt_id=None, custom_prompt_template=None):
        if custom_prompt_template is not None:
            task_prompt = custom_prompt_template.format(query=query)
            task_prompt_template, task_prompt_id = custom_prompt_template, None
        else:
            task_prompt_template_list = rewrite_multi_dialog()
            if task_prompt_id is None:
                task_prompt_id = self.task_prompt_id - 1
            task_prompt_template = task_prompt_template_list[task_prompt_id - 1]
            task_prompt = task_prompt_template.format(query=query)
        return task_prompt_template, task_prompt_id, task_prompt

    def _exception_handler(self, new_query, lang_response):
        if not isinstance(lang_response, str):
            post_processed_results, error_code, error_desc, status = (
                new_query,
                401,
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
        self,
        multi_dialogs,
        task_prompt_id=None,
        custom_prompt_template=None,
        **llm_kwargs,
    ):
        if self._is_params_valid(multi_dialogs):
            new_query = multi_dialogs[-1].get("Query")
            task_prompt_template, task_prompt_id, task_prompt = self._task_prompt(
                multi_dialogs, task_prompt_id, custom_prompt_template
            )
            # 大模型路由
            llm = llm_router(llm_kwargs["scene_name"])
            lang_response = llm(task_prompt, **llm_kwargs)
            post_processed_results, error_code, error_desc, status = (
                self._exception_handler(new_query, lang_response)
            )
            results = {
                "unique_id": unique_id_fn(),
                "r4_module": {"flowCode": self.flowCode, "taskCode": self.taskCode},
                "input": {
                    "queries": multi_dialogs,
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
    rewriteMultiDialog = RewriteMultiDialog()
    # llm_kwargs = {'stream': False, 'top_k': 50, 'top_p': 0.9, 'sceneName': 'tomato_r4_bailing_10b_sst_mft',
    #               'chainName': 'v1', 'modelEnv': 'prod'}
    scene_name = "tomato_r4_bailing_10b_sst_mft"
    llm_kwargs = {"scene_name": scene_name}
    multi_dialog = [
        {"Answer": "云南的新质生产力发展情况"},
        {"Query": "总书记对新质生产力的论述有哪些"},
    ]
    results = rewriteMultiDialog.run(
        multi_dialog, task_prompt_id=5, custom_prompt_template=None, **llm_kwargs
    )
    print(results)
