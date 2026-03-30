"""
dashscope_agent.py
基于阿里云百炼（DashScope）API 的 AIAgent 实现。
使用 OpenAI 兼容接口，默认模型为 qwen-plus。
"""

import os
from typing import Optional
from base_agent import AIAgent
from key_loader import get_api_key


class DashScopeAgent(AIAgent):
    """
    DashScope Agent 实现。

    支持模型示例：
        - qwen-plus
        - qwen-max
        - qwen-turbo
    """

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        agent_name: str = "DashScopeAgent",
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 DashScope Agent。

        Args:
            agent_name: Agent 名称，用于日志标识
            model: 使用的模型，默认 qwen-plus
            api_key: 百炼 API Key；为 None 时从环境变量 DASHSCOPE_API_KEY 读取
            temperature: 生成温度（0~1）
            max_tokens: 单次回复最大 token 数
            system_prompt: 自定义系统提示词；为 None 时使用默认提示词
        """
        resolved_key = api_key or get_api_key("dashscope")
        if not resolved_key:
            raise ValueError(
                f"[{agent_name}] 未提供 API Key，"
                "请设置环境变量 DASHSCOPE_API_KEY，或在项目根目录 API_Key.json 中配置 dashscope_api_key，"
                "或在构造函数中传入 api_key 参数。"
            )

        self._custom_system_prompt = system_prompt

        super().__init__(
            agent_name=agent_name,
            model=model,
            api_key=resolved_key,
            base_url=self.DASHSCOPE_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def build_system_prompt(self) -> str:
        """返回系统提示词；支持构造时自定义传入。"""
        if self._custom_system_prompt:
            return self._custom_system_prompt
        return "你是一个有帮助的 AI 助手，请用中文回答用户的问题。"

    def run(self, user_input: str) -> str:
        """
        执行单轮对话并返回模型回复。

        Args:
            user_input: 用户输入文本

        Returns:
            模型回复的文本内容
        """
        if not self.conversation_history:
            self.create_conversation()
        return self.chat(user_input)


if __name__ == "__main__":
    agent = DashScopeAgent(
        agent_name="百炼测试Agent",
        model="qwen-plus",
        system_prompt="你是一个专业的心理健康顾问，请用简洁的中文回答问题。",
    )

    agent.create_conversation()
    reply = agent.run("你好，请简单介绍一下你自己。")
    print("\n=== 回复 ===")
    print(reply)
