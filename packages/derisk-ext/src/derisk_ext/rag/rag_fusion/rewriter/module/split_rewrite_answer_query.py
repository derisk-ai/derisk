from derisk_ext.rag.rag_fusion.rewriter.module.split_query import SplitQuery
from derisk_ext.rag.rag_fusion.rewriter.module.rewrite_query import RewriteQuery
from derisk_ext.rag.rag_fusion.rewriter.module.answer_query import AnswerQuery


class SplitRewriteAnswer(SplitQuery, RewriteQuery, AnswerQuery):
    def __init__(self):
        super().__init__()
        self.splitQuery = SplitQuery()
        self.rewriteQuery = RewriteQuery()
        self.answerQuery = AnswerQuery()
        self.default_llm_kwargs = {
            "stream": False,
            "top_k": 50,
            "top_p": 0.9,
            # "scene_name": "tomato_r4_bailing_10b_sst_mft",
            "chainName": "v1",
            "modelEnv": "prod",
        }

    def _merge_params(self, custom_params, default_task_prompt_id, scene_name):
        params = {
            "task_prompt_id": custom_params.get(
                "task_prompt_id", default_task_prompt_id
            ),
            "custom_prompt_template": custom_params.get("custom_prompt_template", None),
            "llm_kwargs": {
                **self.default_llm_kwargs,
                **custom_params.get("llm_kwargs", {}),
                "scene_name": scene_name,
            },
        }
        return params

    def run(
        self,
        query,
        split_query_params={},
        rewrite_query_params={},
        answer_query_params={},
        scene_name="tomato_r4_bailing_10b_sst_mft",
    ):
        split_params = self._merge_params(
            split_query_params, self.splitQuery.task_prompt_id, scene_name
        )
        rewrite_params = self._merge_params(
            rewrite_query_params, self.rewriteQuery.task_prompt_id, scene_name
        )
        answer_params = self._merge_params(
            answer_query_params, self.answerQuery.task_prompt_id, scene_name
        )

        split_query_results = self.splitQuery.run(
            query,
            split_params.get("task_prompt_id"),
            split_params.get("custom_prompt_template"),
            **split_params.get("llm_kwargs"),
        )

        rewrite_query_results = self.rewriteQuery.run(
            split_query_results.get("algo_res").get("post_processed_results"),
            rewrite_params.get("task_prompt_id"),
            rewrite_params.get("custom_prompt_template"),
            **rewrite_params.get("llm_kwargs"),
        )

        rewrite_query_post_processed_results = rewrite_query_results.get(
            "algo_res"
        ).get("post_processed_results")
        answer_query_results = self.answerQuery.run(
            rewrite_query_post_processed_results,
            answer_params.get("task_prompt_id"),
            answer_params.get("custom_prompt_template"),
            **answer_params.get("llm_kwargs"),
        )

        module_results = {
            "split_query": split_query_results,
            "rewrite_query": rewrite_query_results,
            "answer_query": answer_query_results,
        }
        fin_results = [
            (i if i is not None else "") + (j if j is not None else "")
            for i, j in zip(
                rewrite_query_post_processed_results,
                answer_query_results.get("algo_res").get("post_processed_results"),
            )
        ]
        results = {"fin_results": fin_results, "module_results": module_results}
        return results


if __name__ == "__main__":
    # query = '总书记在2024年两会上对中国经济前景有何展望？'
    # query = "清明节来啦，在哪里玩耍比较快乐呢"

    query = "embedidng模型"
    # split_query_params = {"task_prompt_id": 1, "llm_kwargs": {"top_p": 0.98}}
    # rewrite_query_params = {
    #     "task_prompt_id": 7,
    #     "llm_kwargs": {"top_p": 0.78},
    #     "custom_prompt_template": """对以下输入一定改写为4个子问题->\n\n输入:{query}\n输出:\n""",
    # }
    splitRewriteAnswer = SplitRewriteAnswer()
    # res = splitRewriteAnswer.run(query, split_query_params, rewrite_query_params)

    # scene_name = "tomato_r4_bailing_10b_sst_mft"
    # scene_name = "Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10"
    scene_name = "Qwen25_14B_Instruct_FP16_vLLM"
    res = splitRewriteAnswer.run(query, scene_name=scene_name)
    print(res)
