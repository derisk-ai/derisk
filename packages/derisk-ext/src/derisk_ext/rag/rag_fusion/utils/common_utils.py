import time
import re
import ast
import shortuuid
import concurrent.futures


def unique_id_fn():
    return shortuuid.uuid()


def execute_parallel_llm(func, args_list, max_workers=10):
    """

    Args:
        func: 执行方法
        args_list: 方法参数列表，大模型为 query、llm_kwargs
        max_workers: 最大并发数

    Returns:
        执行结果列表
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 解包为关键字参数
        future_to_args = {
            executor.submit(func, query=arg[0], **arg[1]): arg for arg in args_list
        }
        results = []
        for future in concurrent.futures.as_completed(future_to_args):
            original_args = future_to_args[future]
            try:
                result = future.result()
                index = args_list.index(original_args)
                results.append((index, result))
            except Exception as e:
                results.append((args_list.index(original_args), f"Error: {e}"))
        results_in_order = sorted(results, key=lambda x: x[0])
        return [result for index, result in results_in_order]


def execute_parallel_retriever(func, args_list, max_workers=10):
    """
    并发执行函数

    Args:
        func: 执行方法
        args_list: 方法参数列表
        max_workers: 最大并发数

    Returns:
        执行结果列表
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 解包为关键字参数
        future_to_args = {executor.submit(func, **arg): arg for arg in args_list}
        results = []
        for future in concurrent.futures.as_completed(future_to_args):
            original_args = future_to_args[future]
            try:
                result = future.result()
                index = args_list.index(original_args)
                results.append((index, result))
            except Exception as e:
                results.append((args_list.index(original_args), f"Error: {e}"))
        results_in_order = sorted(results, key=lambda x: x[0])
        return [result for index, result in results_in_order]


def normalize_to_list(input_item):
    if isinstance(input_item, list):
        new_list = []
        for item in input_item:
            try:
                parsed_item = ast.literal_eval(item)
                if isinstance(parsed_item, list):
                    new_list.extend(parsed_item)
                else:
                    new_list.append(parsed_item)
            except (ValueError, SyntaxError):
                new_list.append(item)
        return new_list
    elif isinstance(input_item, str):
        try:
            list_from_str = ast.literal_eval(input_item)
            if isinstance(list_from_str, list):
                return list_from_str
        except (ValueError, SyntaxError):
            return [input_item]
    else:
        return []


def answer_filter(text):
    if not text:
        return True
    special_char_pattern = re.compile(r"None|对不起|不好意思|我不知道|我没有")
    return bool(special_char_pattern.search(text))


def is_response_error(text):
    pattern = r"\b.*[Ee]rror.*\b"
    return re.search(pattern, text) is not None


def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        # return {'algo_res':result,'elapsed_time':elapsed_time}
        return result, elapsed_time

    return wrapper
