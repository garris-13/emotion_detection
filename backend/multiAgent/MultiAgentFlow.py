"""
MultiAgentFlow.py
多智能体编排流程：情绪数据 → 数据分析师 → 心理评估师 → 行动规划师 → 报告主编 → Markdown 报告
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional

# 确保能找到同目录下的模块
sys.path.insert(0, os.path.dirname(__file__))

from deepseek_agent import DeepSeekAgent
from dashscope_agent import DashScopeAgent
from key_loader import get_api_key

# ------------------------------------------------------------------ #
#  全局配置
# ------------------------------------------------------------------ #

MODEL = "deepseek-chat"


# ------------------------------------------------------------------ #
#  Agent 工厂函数
# ------------------------------------------------------------------ #

def make_agent(name: str, system_prompt: str, temperature: float = 0.7, max_tokens: int = 2000):
    deepseek_key = get_api_key("deepseek")
    if deepseek_key:
        return DeepSeekAgent(
            agent_name=name,
            model=MODEL,
            api_key=deepseek_key,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    # 兼容回退到百炼。
    dashscope_key = get_api_key("dashscope")
    if dashscope_key:
        return DashScopeAgent(
            agent_name=name,
            model="qwen-plus",
            api_key=dashscope_key,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    raise ValueError(
        f"[{name}] 未提供 API Key，请设置环境变量 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY，"
        f"或在项目根目录 API_Key.json 中配置 deepseek_api_key / dashscope_api_key。"
    )


# ------------------------------------------------------------------ #
#  Agent 系统提示词定义
# ------------------------------------------------------------------ #

ANALYST_SYSTEM_PROMPT = """\
你是一个严谨的数据分析师，所有输出必须使用 **Markdown 格式**。
请不要提供任何建议，只用最精炼的语言总结以下情绪数据的分布规律、高频情绪和异常时间点。
输出结构包含：
- ## 情绪分布概览
- ## 高频情绪
- ## 时间维度分析
- ## 异常点标注
"""

PSYCHOLOGIST_SYSTEM_PROMPT = """\
你是一位拥有20年临床经验的心理咨询师，所有输出必须使用 **Markdown 格式**。
请根据数据分析结论，结合用户画像，剖析用户的潜在心理状态。
语气要充满共情与温暖，字数控制在200字以内。
输出结构包含：
- ## 心理状态评估
- ## 潜在成因分析
"""

PLANNER_SYSTEM_PROMPT = """\
你是一个健康生活教练，所有输出必须使用 **Markdown 格式**。
请根据用户的心理评估，提供 **3 个立刻就能执行的微习惯建议**。
要求：具体、可衡量，切忌说教和空泛，每条建议附上执行细节。
输出结构包含：
- ## 行动建议
  - ### 建议一
  - ### 建议二
  - ### 建议三
"""

EDITOR_SYSTEM_PROMPT = """\
你是一位专业的报告主编，所有输出必须使用 **Markdown 格式**。
你的任务是将三位专家（数据分析师、心理评估师、行动规划师）的输出汇总，
按照以下固定模板排版，输出一份美观、结构清晰、适合前端展示的情绪健康报告。

输出模板（严格遵守）：

---
# 🧠 情绪健康分析报告

> 生成时间：{生成时间}  
> 分析样本：{样本数量} 条  
> 用户画像：{用户画像摘要}

---

## 📊 一、数据洞察

{粘贴数据分析师的完整输出}

---

## 💬 二、心理评估

{粘贴心理评估师的完整输出}

---

## 🎯 三、健康行动建议

{粘贴行动规划师的完整输出}

---

## ✅ 四、综合结论

请用 2-3 句话给出整体总结，语气积极，给予用户力量感。

---

*本报告由 EmoCare 多智能体系统自动生成，仅供参考，不构成临床诊断。*
"""


# ------------------------------------------------------------------ #
#  用户画像格式化
# ------------------------------------------------------------------ #

def format_user_context(user_context: dict) -> str:
    age_map = {
        "child": "儿童（<12岁）", "teen": "青少年（12-18岁）",
        "young_adult": "青年（18-30岁）", "adult": "成人（30-60岁）",
        "senior": "老年（>60岁）"
    }
    stress_map = {"low": "低压力", "medium": "中等压力", "high": "高压力"}

    age = age_map.get(user_context.get("age_group", ""), user_context.get("age_group", "未知"))
    stress = stress_map.get(user_context.get("stress_level", ""), user_context.get("stress_level", "未知"))
    support = "有社会支持系统" if user_context.get("has_support_system") else "无社会支持系统"
    experience = "有情绪监测经验" if not user_context.get("is_first_time", True) else "首次使用"
    return f"{age} | {stress} | {support} | {experience}"


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def run_flow(input_path: str, user_context: Optional[dict] = None, output_path: Optional[str] = None) -> str:
    """
    执行多智能体编排流程。

    Args:
        input_path:   input.json 文件路径
        user_context: 用户画像字典，为 None 时使用默认值
        output_path:  报告保存路径（.md），为 None 时自动生成

    Returns:
        最终 Markdown 报告字符串
    """
    # --- 读取输入数据 ---
    print("\n" + "=" * 60)
    print("📂 加载输入数据...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    history_data = raw_data.get("history_data", raw_data)  # 兼容直接传列表或带 key 的结构
    return run_flow_from_data(history_data=history_data, user_context=user_context, output_path=output_path)


def run_flow_from_data(history_data: list, user_context: Optional[dict] = None, output_path: Optional[str] = None) -> str:
    """
    直接使用历史数据执行多智能体流程。

    Args:
        history_data: 历史情绪数据列表
        user_context: 用户画像字典，为 None 时使用默认值
        output_path:  报告保存路径（.md），为 None 时自动生成

    Returns:
        最终 Markdown 报告字符串
    """
    total_samples = len(history_data)

    if total_samples == 0:
        raise ValueError("history_data 不能为空")

    if user_context is None:
        user_context = {
            "age_group": "adult",
            "stress_level": "medium",
            "has_support_system": True,
            "is_first_time": False
        }

    user_context_str = format_user_context(user_context)
    generated_at = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    print(f"✅ 加载完成：{total_samples} 条历史数据，用户画像：{user_context_str}")

    # --- 构建给数据分析师的输入 ---
    analyst_input = (
        f"以下是用户的情绪时间序列数据（共 {total_samples} 条），请进行数据分析：\n\n"
        f"```json\n{json.dumps(history_data, ensure_ascii=False, indent=2)}\n```"
    )

    # ============================
    # Step 1: 数据分析师
    # ============================
    print("\n" + "=" * 60)
    print("🔍 Step 1 / 4 — 数据分析师正在分析数据...")
    analyst_agent = make_agent("数据分析师", ANALYST_SYSTEM_PROMPT, temperature=0.3)
    analyst_agent.create_conversation()
    analyst_output = analyst_agent.run(analyst_input)
    print("✅ 数据分析师完成。")

    # ============================
    # Step 2: 心理评估师
    # ============================
    print("\n" + "=" * 60)
    print("💬 Step 2 / 4 — 心理评估师正在评估...")
    psychologist_input = (
        f"## 数据分析结论\n\n{analyst_output}\n\n"
        f"---\n\n## 用户画像\n\n{user_context_str}"
    )
    psychologist_agent = make_agent("心理评估师", PSYCHOLOGIST_SYSTEM_PROMPT, temperature=0.7, max_tokens=600)
    psychologist_agent.create_conversation()
    psychologist_output = psychologist_agent.run(psychologist_input)
    print("✅ 心理评估师完成。")

    # ============================
    # Step 3: 行动规划师
    # ============================
    print("\n" + "=" * 60)
    print("🎯 Step 3 / 4 — 行动规划师正在制定建议...")
    planner_input = (
        f"## 心理评估结果\n\n{psychologist_output}\n\n"
        f"---\n\n## 用户画像\n\n{user_context_str}"
    )
    planner_agent = make_agent("行动规划师", PLANNER_SYSTEM_PROMPT, temperature=0.5)
    planner_agent.create_conversation()
    planner_output = planner_agent.run(planner_input)
    print("✅ 行动规划师完成。")

    # ============================
    # Step 4: 报告主编（汇总）
    # ============================
    print("\n" + "=" * 60)
    print("📝 Step 4 / 4 — 报告主编正在生成最终报告...")
    editor_input = (
        f"请按照你的模板生成最终报告，以下是各专家的输出和元信息：\n\n"
        f"**生成时间**：{generated_at}\n"
        f"**样本数量**：{total_samples} 条\n"
        f"**用户画像**：{user_context_str}\n\n"
        f"---\n\n"
        f"### 数据分析师输出\n\n{analyst_output}\n\n"
        f"---\n\n"
        f"### 心理评估师输出\n\n{psychologist_output}\n\n"
        f"---\n\n"
        f"### 行动规划师输出\n\n{planner_output}"
    )
    editor_agent = make_agent("报告主编", EDITOR_SYSTEM_PROMPT, temperature=0.5, max_tokens=3000)
    editor_agent.create_conversation()
    final_report = editor_agent.run(editor_input)
    print("✅ 报告主编完成。")

    # --- 保存报告 ---
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"report_{timestamp}.md"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print("\n" + "=" * 60)
    print(f"🎉 流程完成！报告已保存至：{output_path}")
    print("=" * 60 + "\n")

    return final_report


# ------------------------------------------------------------------ #
#  入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), "input.json")

    user_ctx = {
        "age_group": "adult",
        "stress_level": "medium",
        "has_support_system": True,
        "is_first_time": False
    }

    report = run_flow(input_path=input_file, user_context=user_ctx)

    print("\n========== 最终报告预览（前 800 字符）==========")
    print(report[:800])
    print("..." if len(report) > 800 else "")
