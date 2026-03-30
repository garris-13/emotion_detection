"""
base_agent.py
抽象 AIAgent 基类，定义多智能体系统中所有 Agent 的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI


class AIAgent(ABC):
    """
    抽象 AI Agent 基类。

    子类需实现：
        - build_system_prompt(): 构建该 Agent 的系统提示词
        - run(user_input): 执行 Agent 的核心任务并返回结果
    """

    def __init__(
        self,
        agent_name: str,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        初始化 Agent。

        Args:
            agent_name:  Agent 的名称，用于日志标识
            model:       使用的大模型名称
            api_key:     阿里云百炼 API Key；为 None 时从环境变量 DASHSCOPE_API_KEY 读取
            base_url:    兼容 OpenAI 的 API 地址
            temperature: 生成温度（0~1，越高越多样）
            max_tokens:  单次回复最大 token 数
        """
        import os

        self.agent_name = agent_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not resolved_key:
            raise ValueError(
                f"[{agent_name}] 未提供 API Key，"
                "请设置环境变量 DASHSCOPE_API_KEY 或在构造函数中传入 api_key 参数。"
            )

        self.client = OpenAI(api_key=resolved_key, base_url=base_url)
        self.conversation_history: list[dict] = []

    # ------------------------------------------------------------------ #
    #  抽象方法（子类必须实现）
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_system_prompt(self) -> str:
        """
        构建该 Agent 的系统提示词。

        Returns:
            系统提示词字符串
        """

    @abstractmethod
    def run(self, user_input: str) -> str:
        """
        执行 Agent 的核心任务。

        Args:
            user_input: 用户输入或上游 Agent 的输出

        Returns:
            Agent 的处理结果字符串
        """

    # ------------------------------------------------------------------ #
    #  公共方法
    # ------------------------------------------------------------------ #

    def create_conversation(self) -> None:
        """创建（重置）一次新的对话，清空历史消息并注入系统提示词。"""
        self.conversation_history = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        print(f"[{self.agent_name}] 对话已初始化。")

    def add_user_message(self, content: str) -> None:
        """
        向对话历史中追加一条用户消息。

        Args:
            content: 用户消息内容
        """
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """
        向对话历史中追加一条助手消息（用于多轮对话续写）。

        Args:
            content: 助手消息内容
        """
        self.conversation_history.append({"role": "assistant", "content": content})

    def chat(self, user_input: str, *, maintain_history: bool = True) -> str:
        """
        发送用户消息并获取模型回复。

        Args:
            user_input:       本轮用户输入
            maintain_history: 是否将本轮消息追加到历史（多轮对话时置 True）

        Returns:
            模型回复的文本内容
        """
        if not self.conversation_history:
            self.create_conversation()

        if maintain_history:
            self.add_user_message(user_input)
            messages = self.conversation_history
        else:
            # 单轮模式：系统提示词 + 本次用户消息，不修改历史
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": user_input},
            ]

        print(f"[{self.agent_name}] 调用模型 {self.model}，消息轮次：{len(messages)}")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        reply = completion.choices[0].message.content

        if maintain_history:
            self.add_assistant_message(reply)

        print(f"[{self.agent_name}] 回复长度：{len(reply)} 字符，"
              f"消耗 token：{completion.usage.total_tokens if completion.usage else 'N/A'}")

        return reply

    def reset(self) -> None:
        """清空对话历史（不重新注入系统提示词）。"""
        self.conversation_history = []
        print(f"[{self.agent_name}] 对话历史已清空。")

    def get_history(self) -> list[dict]:
        """返回当前对话历史的只读副本。"""
        return list(self.conversation_history)
