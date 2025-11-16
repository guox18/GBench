"""自定义数据解析器示例"""


def parse(raw_data: dict) -> dict:
    """
    将原始数据转换为标准格式
    
    标准格式:
    {
        "id": int,
        "prompt": str,
        "ground_truth": str (可选),
        "metadata": dict (可选),
        "responses": null,
        "judges": null
    }
    
    参数:
        raw_data: 原始数据字典
    
    返回:
        标准格式的数据字典
    """
    return {
        "id": raw_data.get("id", 0),
        "prompt": raw_data.get("question", raw_data.get("prompt", "")),
        "ground_truth": raw_data.get("answer", raw_data.get("ground_truth", "")),
        "metadata": raw_data.get("metadata", {}),
        "responses": None,
        "judges": None,
    }

