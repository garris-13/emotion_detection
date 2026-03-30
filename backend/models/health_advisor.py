"""
情绪健康建议生成器
基于情绪识别结果生成个性化健康建议
"""

# 正确的导入方式
from datetime import datetime  # 从datetime模块导入datetime类
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# 移除之前的 import datetime 语句


@dataclass
class EmotionResult:
    """情绪识别结果"""
    emotion: str  # 主要情绪
    confidence: float  # 置信度
    probabilities: Dict[str, float]  # 所有情绪的概率分布


class HealthAdvisor:
    """
    情绪健康建议生成器

    根据情绪识别结果生成个性化健康建议
    支持规则匹配、概率加权、个性化推荐
    """

    def __init__(self, rules_path: str = None):
        """
        初始化建议生成器

        Args:
            rules_path: 建议规则JSON文件路径
        """
        self.emotion_zh = {
            # "anger": "愤怒",
            # "disgust": "厌恶",
            # "fear": "恐惧",
            # "happy": "快乐",
            # "neutral": "平静",
            # "sad": "悲伤",
            # "surprised": "惊讶"
            'surprised': '惊讶',
            'fear': '恐惧',
            'disgust': '厌恶',
            'happy': '快乐',
            'sad': '悲伤',
            'anger': '愤怒',
            'neutral': '平静'
        }

        # 默认建议规则
        self.default_rules = self._create_default_rules()

        # 加载自定义规则
        self.rules = self.default_rules
        if rules_path:
            self.load_rules(rules_path)

    def _create_default_rules(self) -> Dict:
        """创建默认建议规则"""
        return {
            "anger": {
                "description": "愤怒情绪",
                "risk_level": "medium",
                "immediate_actions": [
                    "深呼吸10次，数到10再行动",
                    "暂时离开当前环境",
                    "喝一杯温水"
                ],
                "daily_tips": [
                    "每天进行15分钟冥想",
                    "练习表达自己的感受而非指责",
                    "尝试写愤怒日记"
                ],
                "long_term_suggestions": [
                    "学习情绪管理技巧",
                    "建立健康的发泄渠道",
                    "考虑心理咨询"
                ],
                "emergency_threshold": 0.8,
                "emergency_advice": "如果感到无法控制愤怒，请立即联系专业人士"
            },
            "disgust": {
                "description": "厌恶情绪",
                "risk_level": "low",
                "immediate_actions": [
                    "转移注意力到美好事物",
                    "听喜欢的音乐",
                    "进行深呼吸"
                ],
                "daily_tips": [
                    "保持环境整洁",
                    "选择积极向上的内容",
                    "培养感恩心态"
                ],
                "long_term_suggestions": [
                    "探索厌恶的根源",
                    "学习接纳不完美",
                    "拓展认知边界"
                ]
            },
            "fear": {
                "description": "恐惧情绪",
                "risk_level": "medium",
                "immediate_actions": [
                    "告诉自己'我很安全'",
                    "进行地面冥想（感受脚下地面）",
                    "拨打信任的人的电话"
                ],
                "daily_tips": [
                    "渐进式暴露疗法",
                    "写恐惧清单并分析",
                    "学习放松技巧"
                ],
                "long_term_suggestions": [
                    "系统脱敏训练",
                    "认知行为疗法",
                    "建立安全支持系统"
                ],
                "emergency_threshold": 0.75,
                "emergency_advice": "如果恐惧严重影响生活，请寻求专业帮助"
            },
            "happy": {
                "description": "快乐情绪",
                "risk_level": "very_low",
                "immediate_actions": [
                    "分享你的快乐给他人",
                    "记录美好时刻",
                    "进行感恩练习"
                ],
                "daily_tips": [
                    "保持积极社交",
                    "培养兴趣爱好",
                    "帮助他人"
                ],
                "long_term_suggestions": [
                    "建立快乐习惯",
                    "学习积极心理学",
                    "成为快乐的传播者"
                ]
            },
            "sad": {
                "description": "悲伤情绪",
                "risk_level": "high",
                "immediate_actions": [
                    "允许自己感受悲伤",
                    "与信任的人倾诉",
                    "进行温和的运动"
                ],
                "daily_tips": [
                    "保持规律作息",
                    "接触阳光和自然",
                    "写情绪日记"
                ],
                "long_term_suggestions": [
                    "建立支持网络",
                    "寻求心理咨询",
                    "学习自我关怀"
                ],
                "emergency_threshold": 0.7,
                "emergency_advice": "如果悲伤持续两周以上或有自杀念头，请立即寻求帮助"
            },
            "surprised": {
                "description": "惊讶情绪",
                "risk_level": "very_low",
                "immediate_actions": [
                    "给自己时间消化",
                    "评估惊讶的性质",
                    "深呼吸保持冷静"
                ],
                "daily_tips": [
                    "培养好奇心",
                    "保持开放心态",
                    "准备应对变化"
                ],
                "long_term_suggestions": [
                    "增强适应能力",
                    "学习危机管理",
                    "培养心理弹性"
                ]
            },
            "neutral": {
                "description": "平静情绪",
                "risk_level": "very_low",
                "immediate_actions": [
                    "享受当下的宁静",
                    "进行简单的深呼吸保持放松",
                    "喝一杯温水"
                ],
                "daily_tips": [
                    "维持良好的生活作息",
                    "记录下今天让你感到平静的事物",
                    "进行适度的散步或锻炼"
                ],
                "long_term_suggestions": [
                    "培养内心的平和与定力",
                    "持续关注自我情绪状态",
                    "发展冥想或正念的习惯"
                ]
            }
        }

    def load_rules(self, rules_path: str):
        """从JSON文件加载规则"""
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                custom_rules = json.load(f)

            # 合并规则
            for emotion, rule in custom_rules.items():
                if emotion in self.rules:
                    self.rules[emotion].update(rule)
                else:
                    self.rules[emotion] = rule

            print(f"已加载自定义规则: {rules_path}")
        except Exception as e:
            print(f"加载规则失败，使用默认规则: {e}")

    def analyze_emotion_pattern(self, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        分析情绪模式

        Args:
            probabilities: 情绪概率字典

        Returns:
            情绪分析结果
        """
        # 找出主要情绪
        main_emotion = max(probabilities.items(), key=lambda x: x[1])

        # 计算情绪复杂度（熵）
        probs = np.array(list(probabilities.values()))
        probs = probs[probs > 0]  # 移除零概率
        if len(probs) > 1:
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(len(probs))
            complexity = entropy / max_entropy
        else:
            complexity = 0.0

        # 情绪强度分析
        intensity = main_emotion[1]

        # 次要情绪
        sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        secondary_emotions = [(e, p) for e, p in sorted_emotions[1:3] if p > 0.1]

        return {
            "main_emotion": main_emotion[0],
            "main_confidence": main_emotion[1],
            "emotion_complexity": float(complexity),
            "intensity_level": self._get_intensity_level(intensity),
            "secondary_emotions": secondary_emotions,
            "has_mixed_emotions": complexity > 0.3,
            "needs_attention": self._needs_attention(probabilities)
        }

    def _get_intensity_level(self, confidence: float) -> str:
        """获取情绪强度等级"""
        if confidence >= 0.8:
            return "非常高"
        elif confidence >= 0.6:
            return "高"
        elif confidence >= 0.4:
            return "中等"
        else:
            return "低"

    def _needs_attention(self, probabilities: Dict[str, float]) -> bool:
        """判断是否需要特别关注"""
        attention_emotions = ["anger", "fear", "sad"]
        for emotion in attention_emotions:
            if probabilities.get(emotion, 0) > 0.6:
                return True
        return False

    def generate_advice(self,
                        emotion_result: EmotionResult,
                        user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        生成个性化健康建议 - 修复JSON序列化问题

        Args:
            emotion_result: 情绪识别结果
            user_context: 用户上下文信息（可选）

        Returns:
            完整的建议报告（JSON可序列化）
        """
        try:
            # 分析情绪模式
            analysis = self.analyze_emotion_pattern(emotion_result.probabilities)

            # 获取基础建议
            main_emotion = analysis["main_emotion"]
            emotion_rule = self.rules.get(main_emotion, {})

            # 个性化调整建议
            personalized_advice = self._personalize_advice(
                emotion_rule, emotion_result, analysis, user_context
            )

            # 检查是否需要紧急建议
            emergency_info = None
            if "emergency_threshold" in emotion_rule:
                if emotion_result.confidence >= emotion_rule["emergency_threshold"]:
                    emergency_info = {
                        "is_emergency": True,  # Python bool
                        "advice": str(emotion_rule.get("emergency_advice", "")),
                        "threshold": float(emotion_rule["emergency_threshold"])
                    }

            # 构建完整报告 - 确保所有值都是JSON可序列化的
            report = {
                "emotion_analysis": {
                    "main_emotion": str(analysis["main_emotion"]),
                    "main_emotion_zh": str(self.emotion_zh.get(main_emotion, main_emotion)),
                    "confidence": float(emotion_result.confidence),
                    "intensity_level": str(analysis["intensity_level"]),
                    "emotion_complexity": float(analysis["emotion_complexity"]),
                    "has_mixed_emotions": bool(analysis["has_mixed_emotions"]),  # 转换为Python bool
                    "secondary_emotions": [
                        {
                            "emotion": str(e),
                            "probability": float(p),
                            "emotion_zh": str(self.emotion_zh.get(e, e))
                        }
                        for e, p in analysis["secondary_emotions"]
                    ]
                },
                "health_advice": personalized_advice,
                "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),  # 使用datetime.now()
                "risk_assessment": {
                    "risk_level": str(emotion_rule.get("risk_level", "unknown")),
                    "needs_attention": bool(analysis["needs_attention"]),  # 转换为Python bool
                    "recommended_action": str("immediate" if analysis["needs_attention"] else "routine")
                }
            }

            # 添加紧急信息
            if emergency_info:
                report["emergency_info"] = emergency_info

            return report

        except Exception as e:
            print(f"❌ 生成健康建议失败: {e}")
            import traceback
            traceback.print_exc()

            # 返回安全的默认报告
            return self._get_safe_default_report(emotion_result)

    def _get_safe_default_report(self, emotion_result: EmotionResult) -> Dict[str, Any]:
        """获取安全的默认报告（JSON可序列化）"""
        return {
            "emotion_analysis": {
                "main_emotion": str(emotion_result.emotion),
                "main_emotion_zh": str(self.emotion_zh.get(emotion_result.emotion, emotion_result.emotion)),
                "confidence": float(emotion_result.confidence),
                "intensity_level": "中等",
                "emotion_complexity": 0.0,
                "has_mixed_emotions": False,
                "secondary_emotions": []
            },
            "health_advice": {
                "description": "基础情绪调节建议",
                "immediate_actions": ["保持平静，深呼吸"],
                "daily_tips": ["记录情绪变化"],
                "long_term_suggestions": ["学习情绪管理"],
                "additional_suggestions": []
            },
            "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),  # 使用datetime.now()
            "risk_assessment": {
                "risk_level": "unknown",
                "needs_attention": False,
                "recommended_action": "routine"
            }
        }

    def _personalize_advice(self, base_rule: Dict,
                            emotion_result: EmotionResult,
                            analysis: Dict,
                            user_context: Optional[Dict]) -> Dict:
        """个性化调整建议"""
        advice = {
            "description": base_rule.get("description", ""),
            "immediate_actions": base_rule.get("immediate_actions", []).copy(),
            "daily_tips": base_rule.get("daily_tips", []).copy(),
            "long_term_suggestions": base_rule.get("long_term_suggestions", []),
            "additional_suggestions": []
        }

        # 根据情绪强度调整
        intensity = emotion_result.confidence
        if intensity > 0.8:
            advice["immediate_actions"].insert(0, "首先确保环境安全")

        # 混合情绪的特殊建议
        if analysis["has_mixed_emotions"]:
            mixed_suggestion = "您可能正在经历复杂情绪，建议："
            if "anger" in analysis["secondary_emotions"]:
                mixed_suggestion += " 给情绪命名，区分不同感受；"
            if "sad" in analysis["secondary_emotions"]:
                mixed_suggestion += " 允许自己完整感受情绪；"
            advice["additional_suggestions"].append(mixed_suggestion)

        # 根据用户上下文调整
        if user_context:
            if user_context.get("is_first_time", False):
                advice["additional_suggestions"].append("这是您第一次使用情绪检测，建议连续记录一周观察模式")

            if user_context.get("has_support_system", True):
                advice["immediate_actions"].append("与信任的人分享您的感受")
            else:
                advice["long_term_suggestions"].append("考虑加入支持小组或建立社交支持网络")

        return advice

    def save_report(self, report: Dict, output_path: str):
        """保存建议报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"报告已保存: {output_path}")
        except Exception as e:
            print(f"保存报告失败: {e}")


def create_advice_from_probabilities(probabilities: Dict[str, float],
                                     confidence: float = None,
                                     rules_path: str = None) -> Dict:
    """
    快速创建建议（简化接口）

    Args:
        probabilities: 情绪概率字典
        confidence: 主情绪置信度（如不提供则自动计算）
        rules_path: 规则文件路径

    Returns:
        建议报告
    """
    # 计算主情绪
    if confidence is None:
        main_emotion = max(probabilities.items(), key=lambda x: x[1])
        confidence = main_emotion[1]
        emotion = main_emotion[0]
    else:
        emotion = max(probabilities, key=probabilities.get)

    # 创建情绪结果对象
    emotion_result = EmotionResult(
        emotion=emotion,
        confidence=confidence,
        probabilities=probabilities
    )

    # 生成建议
    advisor = HealthAdvisor(rules_path)
    report = advisor.generate_advice(emotion_result)

    return report


# 使用示例
if __name__ == "__main__":
    # 示例情绪识别结果
    example_probs = {
        "anger": 0.35,
        "disgust": 0.05,
        "fear": 0.10,
        "happy": 0.15,
        "neural": 0.1,
        "sad": 0.20,
        "surprised": 0.05
    }

    # 生成建议
    report = create_advice_from_probabilities(example_probs)

    print("情绪分析结果:")
    print(json.dumps(report["emotion_analysis"], ensure_ascii=False, indent=2))

    print("\n健康建议:")
    print(f"主要情绪: {report['emotion_analysis']['main_emotion_zh']}")
    print(f"强度: {report['emotion_analysis']['intensity_level']}")
    print("\n立即行动:")
    for i, action in enumerate(report["health_advice"]["immediate_actions"], 1):
        print(f"  {i}. {action}")