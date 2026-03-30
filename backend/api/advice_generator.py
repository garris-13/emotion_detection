"""
API建议生成模块
集成到现有API服务器中
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.health_advisor import HealthAdvisor, EmotionResult
from typing import Dict, Any, Optional


class AdviceAPIGenerator:
    """API建议生成器"""

    def __init__(self, rules_path: str = None):
        self.advisor = HealthAdvisor(rules_path)

    def generate_from_prediction(self,
                                 prediction_result: Dict[str, Any],
                                 user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        从预测结果生成建议

        Args:
            prediction_result: 模型预测结果
            user_context: 用户上下文

        Returns:
            建议报告
        """
        # 提取情绪概率
        if 'probabilities' in prediction_result:
            probabilities = prediction_result['probabilities']
        elif 'emotions' in prediction_result:
            # 兼容不同格式
            probabilities = prediction_result['emotions']
        else:
            raise ValueError("预测结果中未找到情绪概率信息")

        # 创建情绪结果对象
        emotion_result = EmotionResult(
            emotion=prediction_result.get('emotion', 'unknown'),
            confidence=prediction_result.get('confidence', 0.0),
            probabilities=probabilities
        )

        # 生成建议
        report = self.advisor.generate_advice(emotion_result, user_context)

        # 合并原始预测结果
        full_result = {
            "prediction": prediction_result,
            "health_advice_report": report,
            "success": True,
            "message": "成功生成健康建议"
        }

        return full_result