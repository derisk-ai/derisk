import asyncio
import json
import random
import uuid
from typing import Optional

from derisk.util.json_utils import serialize


def final_msg():
    pass


def mock_tasks_str(task_count) -> str:
    tasks = []
    agents = ["容量助手", "sql助手", "元数据助手", "DB流量助手"]
    status = ["init", "running", "success", "error"]

    for i in range(task_count):
        task = {
            "taskId": f"{i}",
            "num": i,
            "parentTask": "0",
            "status": f"{random.choice(status)}",
            "err_msg": "mock",
            "progress": 0,
            "taskContent": "获取当前集群的容量数据",
            "name": "获取当前集群的容量数据",
            "agent": f"{random.choice(agents)}",
            "markdown": "test",
            "link": "",
            "logo": "",
        }
        tasks.append(task)
    return json.dumps(tasks, ensure_ascii=False)


def mock_drsk_plan_str(task_str) -> str:
    think_msg_1 = {"content": "规划思考....(MOCK)"}
    plan_msg1 = {
        "role": "DB容量诊断助手",
        "name": "",
        "logo": "",
        "model": "DeepSeek-R1",
        "markdown": f"```vis-thinking\n{json.dumps(think_msg_1, ensure_ascii=False)}\n```\n根据用户问题当前规划可行步骤如下:\n```drsk-tasks\n{task_str}\n```",
    }
    return json.dumps(plan_msg1, ensure_ascii=False)


def mock_drsk_msg_str(content: str, action_vis: Optional[str] = None) -> str:
    msg1 = [
        {
            "role": "DB容量诊断助手",
            "name": "",
            "model": "DeepSeek-R1",
            "logo": "",
            "content": f"{content}",
            "markdown": f"```vis-thinking\nthinking...\n```\n{action_vis}",
            "sender": "test\n```code\n import os \n```",
            "receiver": "receiver",
        }
    ]
    return json.dumps(msg1, ensure_ascii=False)


def mock_drsk_tool_run(tool_name, tool_result):
    status = ["init", "running", "success", "error"]
    param = {
        "name": tool_name,
        "args": {"key1": "value1", "key2": {"kk1": "vv1", "kk2": "vv2"}},
        "status": f"{random.choice(status)}",
        "logo": None,
        "result": tool_result,
        "err_msg": "",
    }
    return f"```vis-plugin:test999:all\n{json.dumps(param, ensure_ascii=False)}\n"


async def t_vis_protocol():
    # mock chunk 1
    chunk_1 = f"""
    ```drsk-plan-msg:test1231:all\n{mock_drsk_plan_str(mock_tasks_str(3))}\n```
    """
    yield f"data:{chunk_1}\n\n"
    await asyncio.sleep(1)

    # mock chunk 2
    chunk_2 = f"""
    ```drsk-msg:test4561:incr\n{mock_drsk_msg_str("增量流式输出测试1")}\n```
    """
    yield f"data:{chunk_2}\n\n"
    await asyncio.sleep(1)

    # mock chunk 3
    chunk_3 = f"""
    ```drsk-msg:test4561:incr\n{mock_drsk_msg_str("23456")}\n```
    """
    yield f"data:{chunk_3}\n\n"
    await asyncio.sleep(1)

    # mock chunk 4
    chunk_4 = f"""
    ```drsk-msg:test4561:incr\n{mock_drsk_msg_str(",流式测试结束")}\n```
    """
    yield f"data:{chunk_4}\n\n"
    await asyncio.sleep(1)

    # mock final chunk
    final_chunk = f"""
    ```vis-plan\n{mock_drsk_plan_str(mock_tasks_str(3))}\n```
    ```drsk-msg:test4561:incr\n{mock_drsk_msg_str("目标任务1，模型思考..", mock_drsk_tool_run("get_db_capacity", '{"value":"0.7"}'))}\n```
    ```drsk-msg:test4562:incr\n{mock_drsk_msg_str("目标任务2，模型思考..", mock_drsk_tool_run("get_qtps)", '[{"time":123456,"value":"0.7"},{"time":123461,"value":"1.7"}]'))}\n```
    ```drsk-plan-msg:test1232:all\n{mock_drsk_plan_str(mock_tasks_str(1))}\n```
    ```drsk-msg:test4562:incr\n{mock_drsk_msg_str("目标任务3，模型思考..", mock_drsk_tool_run("get_change)", '[{"time":123456,"value":"0.7"},{"time":123461,"value":"1.7"}]'))}\n```
    ```drsk-report:test789:incr\n{json.dumps(mock_drsk_msg_str("最终结论报告如下:1111111111111111111111"), ensure_ascii=False)}\n```
    """
    yield f"data:{final_chunk}\n\n"


async def t_vis_protocol_sq():
    # mock final chunk

    final_chunk = f"""123\n```vis-thinking\nthinking...\n```\n```agent-plans\n{mock_tasks_str(3)}\n```\n```agent-messages\n{mock_drsk_msg_str("目标任务1，模型思考..", mock_drsk_tool_run("get_db_capacity", '{"value":"0.7"}'))}\n```\n```agent-messages\n{mock_drsk_msg_str("目标任务2，模型思考..", mock_drsk_tool_run("get_qtps)", '[{"time":123456,"value":"0.7"},{"time":123461,"value":"1.7"}]'))}\n```\n```agent-plans\n{mock_drsk_plan_str(mock_tasks_str(3))}\n```"""

    task_chunk = f"""```agent-plans\n{mock_tasks_str(3)}\n```"""
    test_chunk = f"""```agent-messages\n{mock_drsk_msg_str("目标任务1，模型思考..", mock_drsk_tool_run("get_db_capacity", '{"value":"0.7"}'))}\n```"""
    content = json.dumps({"vis": final_chunk}, default=serialize, ensure_ascii=False)
    content = content.replace("\n", "\\n")
    yield f"data:{content}\n\n"

    # test\n
    # ```agent-plans\n{mock_drsk_plan_str(mock_tasks_str(3))}\n```

    # ```drsk-plan-msg\n{mock_drsk_plan_str(mock_tasks_str(1))}\n```
    # ```agent-messages\n{mock_drsk_msg_str("目标任务3，模型思考..", mock_drsk_tool_run("get_change)", '[{"time":123456,"value":"0.7"},{"time":123461,"value":"1.7"}]'))}\n```
    # ```json\n{json.dumps(mock_drsk_msg_str("最终结论报告如下:1111111111111111111111"))}\n```

    # from derisk.vis import VisAgentMessages
    # vis = VisAgentMessages()
    # chunk = await vis.display(content=[{"sender": "test",
    #                                    "receiver": "",
    #                                    "model": "deepseek",
    #                                    "markdown": "test",
    #                                    }])
    #
    # content = json.dumps({"vis": chunk}, default=serialize, ensure_ascii=False)
    # content = content.replace("\n", "\\n")
    # yield f"data:{content} \n\n"
    #


async def derisk_vis():
    async for msg in t_vis_protocol():
        msg = msg.replace("\n", "\\n")
        print(msg)


if __name__ == "__main__":
    asyncio.run(derisk_vis())
