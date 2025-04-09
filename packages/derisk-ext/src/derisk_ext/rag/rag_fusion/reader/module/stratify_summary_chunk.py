from derisk_ext.rag.rag_fusion.llm.llm_route import llm_router
from derisk_ext.rag.rag_fusion.utils.common_utils import (
    unique_id_fn,
    normalize_to_list,
    answer_filter,
    execute_parallel_llm,
)
from derisk_ext.rag.rag_fusion.reader.task_prompt_template_pool.stratify_summary_chunk import (
    stratify_summary_chunk,
)


class StratifySummaryChunk:
    def __init__(self):
        self.flowCode = "reader"
        self.taskCode = "stratify_summary_chunk"
        self.task_prompt_id = 2

    def _task_prompt(
        self, query, document, task_prompt_id=None, custom_prompt_template=None
    ):
        if custom_prompt_template is not None:
            task_prompt = custom_prompt_template.format(query=query, document=document)
            task_prompt_template, task_prompt_id = custom_prompt_template, None
        else:
            # task_prompt_template_list = reader_prompt_template_pool.get(self.taskCode)
            task_prompt_template_list = stratify_summary_chunk()
            if task_prompt_id is None:
                task_prompt_id = self.task_prompt_id - 1
            task_prompt_template = task_prompt_template_list[task_prompt_id - 1]
            # print(task_prompt_template)
            task_prompt = task_prompt_template.format(query=query, document=document)
        return task_prompt_template, task_prompt_id, task_prompt

    @staticmethod
    def _exception_handler(documents, lang_response):
        lang_response = normalize_to_list(lang_response)
        if not isinstance(lang_response, list):
            post_processed_results, error_code, error_desc, status = (
                documents,
                501,
                "model return incorrect format",
                "failed",
            )
        else:
            post_processed_results, error_code, error_desc, status = (
                [item if not answer_filter(item) else "" for item in lang_response],
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
        documents = normalize_to_list(documents)
        task_prompt_list = list()
        for document in documents:
            task_prompt_template, task_prompt_id, task_prompt = self._task_prompt(
                query, document, task_prompt_id, custom_prompt_template
            )
            task_prompt_list.append(task_prompt)
        args_list = [(task_prompt, llm_kwargs) for task_prompt in task_prompt_list]
        # 大模型路由
        llm = llm_router(llm_kwargs["scene_name"])
        lang_response = execute_parallel_llm(llm, args_list, max_workers=10)
        # print(f'lang_response is: {lang_response}')
        post_processed_results, error_code, error_desc, status = (
            self._exception_handler(query, lang_response)
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
                # list(set(post_processed_results)),
            },
        }
        return results


if __name__ == "__main__":
    stratifySummaryChunk = StratifySummaryChunk()
    # llm_kwargs = {'stream': False, 'top_k': 40, 'top_p': 0.98, 'sceneName': 'tomato_r4_bailing_10b_sst_mft',
    #               'chainName': 'v1', 'modelEnv': 'prod'}

    scene_name = "Qwen25_14B_Instruct_FP16_vLLM"
    llm_kwargs = {"scene_name": scene_name}

    query = "大模型在城市交通方面有哪些应用呢"
    documents = [
        "大模型技术的出现为智慧城市建设带来新的发展变化，特别在城市规划、政务服务、交通管理、环境监测、公共安全等重要领域提供了有效的助力，大大推动了智慧城市建设的步伐。",
        "未来大模型在智慧城市应用场景将更加广泛。除了现有的应用领域（如智能交通管控和环境监测）之外，大模型还将涉足更多新的应用场景。例如，智能停车系统可以利用大模型分析城市中的停车需求和停车资源分布，从而实现更高效的停车引导和资源管理。智能物流领域也能受益于大模型的应用，通过分析实时的物流数据，优化货物运输路径和配送计划，降低运营成本和能源消耗。",
        "AI大模型正加速第三次“数实融合”浪潮全面到来，智能化是其主要特征。AI大模型将影响制造业发展格局，AI大模型将会融入制造业的研发设计、生产工艺、质量管理、运营控制、营销服务、组织协同和经营管理的方方面面。",
        "在生产制造环节，AI大模型可以直接服务智能汽车、机器人、芯片、服装等产品的研发创新，例如工程师可通过大模型自动生成代码指令，完成机器人功能的开发与调试，甚至还能为机器人创造一些全新的功能。",
        "针对现阶段人工智能治理的局限性，可从三方面采取措施。一是充分评估数据和模型的伦理影响和风险点；二是从流程方面，不仅要考虑通用模型，还要把通用模型纳入到社会生态系统、社会政治经济系统来考虑；三是探索伦理方法，充分探索“价值敏感设计”“负责任创新”等伦理方法在通用模型语境下的可行性。",
        "神经网络模型的可解释性，事实性，因果推理能力等问题，这些问题并不能随着神经网络规模的扩大而解决，可能将会成为长期成为困扰这一技术方向的问题。",
    ]
    results = stratifySummaryChunk.run(query, documents, **llm_kwargs)
    print(results)
    # results = [
    #     '大模型技术的出现为智慧城市建设带来新的发展变化，特别在城市规划、政务服务、交通管理、环境监测、公共安全等重要领域提供了有效的助力。',
    #     '智能交通管控和环境监测是现有的大模型应用领域。此外，智能停车系统可以利用大模型分析城市中的停车需求和停车资源分布，从而实现更高效的停车引导和资源管理。',
    #     '', '', '', '']
