"""
表情识别 API 客户端
用于调用表情识别 API
包含健康建议功能
"""

import requests
import base64
from PIL import Image
import io
import json
import os


class EmotionRecognitionClient:
    """表情识别 API 客户端"""

    def __init__(self, base_url='http://localhost:5000'):
        """
        初始化客户端

        Args:
            base_url: API 服务器地址
        """
        self.base_url = base_url

    def check_health(self):
        """检查 API 健康状态"""
        try:
            response = requests.get(f'{self.base_url}/health')
            return response.json()
        except Exception as e:
            return {'error': str(e)}

    def get_emotions(self):
        """获取支持的表情列表"""
        try:
            response = requests.get(f'{self.base_url}/emotions')
            return response.json()
        except Exception as e:
            return {'error': str(e)}

    def predict_from_file(self, image_path):
        """
        从图像文件预测表情

        Args:
            image_path: 图像文件路径

        Returns:
            dict: 预测结果
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f'{self.base_url}/predict', files=files)
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_from_base64(self, base64_str):
        """
        从base64编码的图像预测表情

        Args:
            base64_str: base64 编码的图像字符串

        Returns:
            dict: 预测结果
        """
        try:
            headers = {'Content-Type': 'application/json'}
            data = {'image': base64_str}
            response = requests.post(f'{self.base_url}/predict',
                                     json=data, headers=headers)
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_from_pil(self, pil_image):
        """
        从PIL Image对象预测表情

        Args:
            pil_image: PIL Image 对象

        Returns:
            dict: 预测结果
        """
        try:
            # 将PIL图像转换为base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return self.predict_from_base64(img_str)
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_batch(self, image_paths):
        """
        批量预测表情

        Args:
            image_paths: 图像文件路径列表

        Returns:
            dict: 预测结果
        """
        try:
            files = [('images', open(path, 'rb')) for path in image_paths]
            response = requests.post(f'{self.base_url}/predict_batch', files=files)

            # 关闭文件
            for _, f in files:
                f.close()

            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ================ 新增：健康建议相关方法 ================

    def predict_with_advice(self, image_path, user_context=None):
        """
        预测情绪并获取健康建议

        Args:
            image_path: 图像文件路径
            user_context: 用户上下文信息（字典），可选参数，如：
                {
                    "age_group": "adult",  # 年龄组：child, teen, adult, elder
                    "gender": "male",      # 性别：male, female, other
                    "has_support_system": True,  # 是否有支持系统
                    "is_first_time": False,      # 是否首次使用
                    "previous_emotions": [],     # 历史情绪记录
                    "stress_level": "medium"     # 压力水平：low, medium, high
                }

        Returns:
            dict: 包含预测结果和健康建议的完整报告
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}

                data = {}
                if user_context:
                    data['user_context'] = json.dumps(user_context)

                response = requests.post(f'{self.base_url}/predict_with_advice',
                                         files=files, data=data)

            return response.json()

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def analyze_emotion_pattern(self, probabilities, confidence=None, user_context=None):
        """
        直接分析情绪概率模式

        Args:
            probabilities: 情绪概率字典，如：
                {
                    "anger": 0.65,
                    "disgust": 0.05,
                    "fear": 0.10,
                    "happy": 0.05,
                    "sad": 0.10,
                    "surprised": 0.05
                }
            confidence: 主情绪置信度，如不提供则自动计算
            user_context: 用户上下文信息（字典）

        Returns:
            dict: 情绪分析报告
        """
        try:
            url = f'{self.base_url}/advice/analysis'

            data = {
                'probabilities': probabilities
            }

            if confidence is not None:
                data['confidence'] = confidence

            if user_context:
                data['user_context'] = user_context

            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=data, headers=headers)

            return response.json()

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_advice_rules(self):
        """
        获取当前使用的建议规则

        Returns:
            dict: 建议规则信息
        """
        try:
            response = requests.get(f'{self.base_url}/advice/rules')
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}


def print_result(result):
    """打印预测结果"""
    if result.get('success'):
        print(f"\n✅ 预测成功")
        print(f"表情: {result.get('emotion_zh', result.get('emotion', '未知'))}")
        print(f"置信度: {result.get('confidence', 0):.2%}")
        print(f"\n各类别概率:")
        for emotion, prob in result.get('probabilities', {}).items():
            bar = '█' * int(prob * 50)
            print(f"  {emotion:12s}: {bar} {prob:.2%}")
    else:
        print(f"\n❌ 预测失败: {result.get('error', '未知错误')}")


def print_advice_report(report):
    """打印健康建议报告"""
    if not report.get('success', True):
        print(f"❌ 报告生成失败: {report.get('error', '未知错误')}")
        return

    print("\n" + "=" * 70)
    print("🧠 情绪健康分析报告")
    print("=" * 70)

    # 情绪分析部分
    analysis = report.get('emotion_analysis', {})
    if analysis:
        print(f"📊 情绪分析:")
        print(f"   主要情绪: {analysis.get('main_emotion_zh', '未知')}")
        print(f"   置信度: {analysis.get('confidence', 0):.2%}")
        print(f"   强度等级: {analysis.get('intensity_level', '未知')}")
        print(f"   情绪复杂度: {analysis.get('emotion_complexity', 0):.3f}")

        # 次要情绪
        secondary = analysis.get('secondary_emotions', [])
        if secondary:
            print(f"   次要情绪: ", end="")
            for i, sec in enumerate(secondary):
                if i > 0:
                    print(", ", end="")
                print(f"{sec.get('emotion_zh', sec.get('emotion', '未知'))}({sec.get('probability', 0):.1%})", end="")
            print()

    # 健康建议部分
    advice = report.get('health_advice', {})
    if advice:
        print(f"\n💡 健康建议:")
        print(f"   描述: {advice.get('description', '')}")

        # 立即行动
        immediate = advice.get('immediate_actions', [])
        if immediate:
            print(f"\n   🚨 立即行动:")
            for i, action in enumerate(immediate, 1):
                print(f"      {i}. {action}")

        # 日常贴士
        daily = advice.get('daily_tips', [])
        if daily:
            print(f"\n   📅 日常贴士:")
            for i, tip in enumerate(daily, 1):
                print(f"      {i}. {tip}")

        # 长期建议
        long_term = advice.get('long_term_suggestions', [])
        if long_term:
            print(f"\n   🌱 长期建议:")
            for i, suggestion in enumerate(long_term, 1):
                print(f"      {i}. {suggestion}")

        # 额外建议
        additional = advice.get('additional_suggestions', [])
        if additional:
            print(f"\n   📝 额外建议:")
            for i, suggestion in enumerate(additional, 1):
                print(f"      {i}. {suggestion}")

    # 风险评估
    risk = report.get('risk_assessment', {})
    if risk:
        print(f"\n⚠️  风险评估:")
        risk_level = risk.get('risk_level', 'unknown')
        risk_colors = {
            'very_low': '🟢',
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'very_high': '🔴'
        }
        color = risk_colors.get(risk_level, '⚪')
        print(f"   风险等级: {color} {risk_level}")
        print(f"   是否需要关注: {'是' if risk.get('needs_attention', False) else '否'}")
        print(f"   建议行动: {risk.get('recommended_action', 'routine')}")

    # 紧急信息
    emergency = report.get('emergency_info', {})
    if emergency and emergency.get('is_emergency', False):
        print(f"\n🚨 紧急提示:")
        print(f"   ❗ {emergency.get('advice', '')}")

    print("\n" + "=" * 70)
    print(f"📅 报告时间: {report.get('timestamp', '未知')}")


def print_full_advice_result(result):
    """打印完整的预测+建议结果"""
    if not result.get('success', True):
        print(f"❌ 请求失败: {result.get('error', '未知错误')}")
        return

    # 打印预测结果
    prediction = result.get('prediction', {})
    if prediction:
        print_result(prediction)

    # 打印健康建议
    advice_report = result.get('health_advice_report', {})
    if advice_report:
        print_advice_report(advice_report)


# 示例用法
if __name__ == '__main__':
    # 创建客户端
    client = EmotionRecognitionClient('http://localhost:5000')

    print("=" * 70)
    print("🧠 表情识别与健康建议 API 客户端")
    print("=" * 70)

    # 1. 检查健康状态
    print("\n1. 检查 API 健康状态...")
    health = client.check_health()
    print(f"状态: {health}")

    # 2. 获取支持的表情
    print("\n2. 获取支持的表情列表...")
    emotions = client.get_emotions()
    print(f"支持的表情: {emotions}")

    # 3. 获取建议规则
    print("\n3. 获取健康建议规则...")
    rules = client.get_advice_rules()
    if rules.get('success'):
        print(f"已加载 {rules.get('count', 0)} 条情绪建议规则")
        for emotion, info in rules.get('rules', {}).items():
            print(f"  - {emotion}: {info.get('description', '无描述')} (风险: {info.get('risk_level', '未知')})")

    # 4. 示例：直接分析情绪概率
    print("\n4. 示例：分析情绪概率模式...")
    example_probs = {
        "anger": 0.45,
        "disgust": 0.05,
        "fear": 0.10,
        "happy": 0.15,
        "sad": 0.20,
        "surprised": 0.05
    }

    user_context = {
        "age_group": "adult",
        "has_support_system": True,
        "is_first_time": True
    }

    analysis_result = client.analyze_emotion_pattern(
        probabilities=example_probs,
        user_context=user_context
    )

    if analysis_result.get('success'):
        print_advice_report(analysis_result.get('report', {}))
    else:
        print(f"分析失败: {analysis_result.get('error', '未知错误')}")

    # 5. 测试图像预测+建议（如果有测试图像）
    test_image_path = 'test_image.jpg'
    if os.path.exists(test_image_path):
        print(f"\n5. 测试图像预测+健康建议: {test_image_path}")

        user_context = {
            "age_group": "adult",
            "gender": "male",
            "has_support_system": True,
            "is_first_time": False,
            "stress_level": "medium"
        }

        full_result = client.predict_with_advice(test_image_path, user_context)
        print_full_advice_result(full_result)
    else:
        print(f"\n5. 跳过图像预测测试（未找到测试图像: {test_image_path}）")

    print("\n" + "=" * 70)
    print("✅ 客户端测试完成")
    print("=" * 70)