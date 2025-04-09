from derisk_ext.rag.rag_fusion.llm.llm_route import llm_router
from derisk_ext.rag.rag_fusion.utils.common_utils import unique_id_fn
from derisk_ext.rag.rag_fusion.reader.task_prompt_template_pool.stratify_summary_gather import (
    stratify_summary_gather,
)


class StratifySummaryGather:
    def __init__(self):
        self.flowCode = "reader"
        self.taskCode = "stratify_summary_gather"
        self.task_prompt_id = 3

    def _task_prompt(
        self, query, documents, task_prompt_id=None, custom_prompt_template=None
    ):
        if custom_prompt_template is not None:
            task_prompt = custom_prompt_template.format(
                query=query, documents=documents
            )
            task_prompt_template, task_prompt_id = custom_prompt_template, None
        else:
            # task_prompt_template_list = reader_prompt_template_pool.get(self.taskCode)
            task_prompt_template_list = stratify_summary_gather()
            if task_prompt_id is None:
                task_prompt_id = self.task_prompt_id - 1
            else:
                task_prompt_id = task_prompt_id - 1
            task_prompt_template = task_prompt_template_list[task_prompt_id - 1]
            task_prompt = task_prompt_template.format(query=query, documents=documents)
        return task_prompt_template, task_prompt_id, task_prompt

    @staticmethod
    def _exception_handler(lang_response):
        if not isinstance(lang_response, str):
            post_processed_results, error_code, error_desc, status = (
                lang_response,
                601,
                ("model return incorrect format"),
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
        query,
        documents,
        task_prompt_id=None,
        custom_prompt_template=None,
        **llm_kwargs,
    ):
        task_prompt_template, task_prompt_id, task_prompt = self._task_prompt(
            query, documents, task_prompt_id, custom_prompt_template
        )
        # 大模型路由
        llm = llm_router(llm_kwargs["scene_name"])
        lang_response = llm(task_prompt, **llm_kwargs)
        post_processed_results, error_code, error_desc, status = (
            self._exception_handler(lang_response)
        )
        results = {
            "unique_id": unique_id_fn(),
            "r4_module": {"flowCode": self.flowCode, "taskCode": self.taskCode},
            "input": {
                "query": query,
                "documents": documents,
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
    stratifySummaryGather = StratifySummaryGather()
    # llm_kwargs = {'stream': False, 'top_k': 40, 'top_p': 0.98, 'sceneName': 'tomato_r4_bailing_10b_sst_mft',
    #               'chainName': 'v1', 'modelEnv': 'prod'}
    query = "大模型在城市交通方面有哪些应用呢"
    documents = [
        "大模型技术的出现为智慧城市建设带来新的发展变化，特别在城市规划、政务服务、交通管理、环境监测、公共安全等重要领域提供了有效的助力。",
        "智能交通管控和环境监测是现有的大模型应用领域。此外，智能停车系统可以利用大模型分析城市中的停车需求和停车资源分布，从而实现更高效的停车引导和资源管理。",
        "",
        "",
        "",
        "",
    ]
    # scene_name = 'Qwen25_14B_Instruct_FP16_vLLM'
    scene_name = "Qwen_2_5_72B_32K_Chat_FP16_vLLM_8A10"
    llm_kwargs = {"scene_name": scene_name}

    results = stratifySummaryGather.run(query, documents, **llm_kwargs)
    print(results)
    # results = ('大模型在城市交通方面的应用主要表现在以下几个方面：\n\n 1. 智能交通管控：大模型技术可以协助进行交通流量预测、道路优化设计，以及实时路况监控等，以提高交通管理效率。\n\n 2. '
    #            '环境监测：通过大模型，能进行空气质量、道路污染等环境因素的监测，有助于环保政策的制定与执行。\n\n 3. '
    #            '智能停车系统：大模型能分析城市中的停车需求和停车资源分布，从而实现更高效的停车引导和资源管理。\n\n '
    #            '总的来说，大模型技术为智慧城市的交通管理带来了新的发展变化，提高了城市交通管理的效率，并改善了环境质量。')
