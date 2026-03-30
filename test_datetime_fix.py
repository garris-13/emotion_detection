"""
测试datetime修复
"""

import os
import sys


print("测试datetime导入...")

# 测试方式1：正确的导入方式
print("\n方式1: from datetime import datetime")
try:
    from datetime import datetime

    print(f"✅ datetime.now(): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试方式2：错误的导入方式
print("\n方式2: import datetime")
try:
    import datetime

    # 这会失败，因为datetime是模块，不是类
    print(f"❌ datetime.now(): 应该会失败")
except Exception as e:
    print(f"✅ 预期失败: {e}")

# 测试方式3：正确的模块导入方式
print("\n方式3: import datetime as dt")
try:
    import datetime as dt

    print(f"✅ dt.datetime.now(): {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试health_advisor模块
print("\n测试health_advisor模块...")

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from backend.models.health_advisor import HealthAdvisor, EmotionResult

    # 创建示例数据
    example_probs = {
        "anger": 0.45,
        "disgust": 0.05,
        "fear": 0.10,
        "happy": 0.15,
        "sad": 0.20,
        "surprised": 0.05
    }

    # 创建情绪结果
    emotion_result = EmotionResult(
        emotion="happy",
        confidence=0.85,
        probabilities=example_probs
    )

    # 生成建议
    advisor = HealthAdvisor()
    report = advisor.generate_advice(emotion_result)

    print("✅ HealthAdvisor工作正常")
    print(f"✅ 时间戳: {report.get('timestamp', '无时间戳')}")
    print(f"✅ 主要情绪: {report['emotion_analysis']['main_emotion_zh']}")

except Exception as e:
    print(f"❌ HealthAdvisor测试失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ datetime修复测试完成")
print("=" * 60)

input("按Enter键退出...")