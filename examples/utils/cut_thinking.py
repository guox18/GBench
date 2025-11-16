"""响应后处理：移除思考过程"""


def process(response: str) -> str:
    """
    从响应中移除 thinking 部分，保留最终答案

    支持多种格式：
    - <think>...</think> 格式（R1 等模型）
    - <|channel|>analysis<|message|> 格式（GPT-OSS 等模型）

    参数:
        response: 模型生成的原始响应

    返回:
        处理后的响应文本
    """
    # 处理标准 <think>...</think> 格式
    if "<think>" in response:
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        else:
            # 输出被截断，返回空行
            response = "\n"
    else:
        response = response.strip()

    # 处理 GPT OSS 格式的 analysis channel
    if "<|channel|>analysis<|message|>" in response:
        if "<|end|><|start|>assistant<|channel|>final<|message|>" in response:
            response = response.split(
                "<|end|><|start|>assistant<|channel|>final<|message|>"
            )[-1].strip()
        else:
            # 输出被截断，返回空行
            response = "\n"
    else:
        response = response.strip()

    # 兜底处理：某些 chat template 不包含开始标签，直接根据 </think> 截断
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response

