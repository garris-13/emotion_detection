"""
deepseek_agent.py
基于 DeepSeek API 的 AIAgent 实现。
DeepSeek API 兼容 OpenAI 格式，直接复用 base_agent 中的 OpenAI 客户端。
"""

import os
from typing import Optional
from base_agent import AIAgent
from key_loader import get_api_key


class DeepSeekAgent(AIAgent):
    """
    DeepSeek Agent 实现。

    支持模型：
        - deepseek-chat     → DeepSeek-V3（非思考模式）
        - deepseek-reasoner → DeepSeek-V3（思考模式）
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        agent_name: str = "DeepSeekAgent",
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 DeepSeek Agent。

        Args:
            agent_name:    Agent 名称，用于日志标识
            model:         使用的模型，可选 "deepseek-chat" 或 "deepseek-reasoner"
            api_key:       DeepSeek API Key；为 None 时从环境变量 DEEPSEEK_API_KEY 读取
            temperature:   生成温度（0~1）
            max_tokens:    单次回复最大 token 数
            system_prompt: 自定义系统提示词；为 None 时使用默认提示词
        """
        resolved_key = api_key or get_api_key("deepseek")
        if not resolved_key:
            raise ValueError(
                f"[{agent_name}] 未提供 API Key，"
                "请设置环境变量 DEEPSEEK_API_KEY，或在项目根目录 API_Key.json 中配置 deepseek_api_key，"
                "或在构造函数中传入 api_key 参数。"
            )

        self._custom_system_prompt = system_prompt

        super().__init__(
            agent_name=agent_name,
            model=model,
            api_key=resolved_key,
            base_url=self.DEEPSEEK_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------ #
    #  实现抽象方法
    # ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  简单验证入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    agent = DeepSeekAgent(
        agent_name="测试Agent",
        model="deepseek-chat",
        system_prompt="你是一个专业的心理健康顾问，请用简洁的中文回答问题。",
    )

    agent.create_conversation()

    # 第一轮
    reply1 = agent.run("你好，请简单介绍一下你自己。")
    print("\n=== 回复 ===")
    print(reply1)

    # 第二轮（多轮对话，保留上下文）
    reply2 = agent.chat("我最近压力很大，有什么缓解建议吗？")
    print("\n=== 回复 ===")
    print(reply2)

    # 查看对话历史
    print(f"\n=== 对话历史（共 {len(agent.get_history())} 条消息） ===")
    for msg in agent.get_history():
        role = "系统" if msg["role"] == "system" else ("用户" if msg["role"] == "user" else "助手")
        print(f"[{role}]: {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
