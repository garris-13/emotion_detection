"""
健康建议使用示例
"""

import sys
import os

# 确保项目根目录在 sys.path 中（便于直接从 examples 目录运行示例）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.api.api_client import EmotionRecognitionClient


def main():
    # 创建客户端
    client = EmotionRecognitionClient('http://localhost:7860')

    # 示例1：直接使用情绪概率
    print("示例1：基于情绪概率生成建议")
    example_probs = {
        "anger": 0.65,
        "disgust": 0.05,
        "fear": 0.10,
        "happy": 0.05,
        "sad": 0.10,
        "surprised": 0.05
    }

    analysis = client.analyze_emotion_pattern(example_probs)
    if analysis['success']:
        report = analysis['report']
        print(f"主要情绪: {report['emotion_analysis']['main_emotion_zh']}")
        print(f"情绪强度: {report['emotion_analysis']['intensity_level']}")
        print(f"风险等级: {report['risk_assessment']['risk_level']}")
        print("\n立即行动建议:")
        for i, action in enumerate(report['health_advice']['immediate_actions'], 1):
            print(f"  {i}. {action}")

    # 示例2：图像预测+建议
    print("\n" + "=" * 50)
    print("示例2：图像情绪识别+健康建议")

    # 替换为实际图像路径
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        user_context = {
            "is_first_time": False,
            "has_support_system": True,
            "age_group": "adult"
        }

        result = client.predict_with_advice(image_path, user_context)

        if result['success']:
            pred = result['prediction']
            advice = result['health_advice_report']

            print(f"识别结果: {pred['emotion_zh']} ({pred['confidence']:.2%})")
            print(f"综合建议: {advice['health_advice']['description']}")

            if advice.get('emergency_info', {}).get('is_emergency'):
                print(f"⚠️ 紧急建议: {advice['emergency_info']['advice']}")


if __name__ == "__main__":
    main()