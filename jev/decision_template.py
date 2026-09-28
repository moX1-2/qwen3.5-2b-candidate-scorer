"""候选评分专用图文模板。直接构造字符串，不执行聊天 Jinja。"""


def render_image_decision(processor, branch_text):
    tokens = [processor.vision_start_token, processor.image_token, processor.vision_end_token]
    if any(token in branch_text for token in tokens):
        raise ValueError('题目或候选包含保留图像标记')
    # 图像占位符由原生 Processor 按 image_grid_thw 展开。
    # 不加入聊天角色、消息结束符或 assistant 生成提示。
    return ''.join(tokens) + '\n' + branch_text
