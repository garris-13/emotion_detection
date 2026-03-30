"""
测试多模态智能 Agent 功能
"""
import os
import sys
from datetime import datetime, timedelta

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backend', 'api'))

# 导入 Agent
from emotion_agent import get_agent

def test_agent():
    print("=" * 70)
    print("🧠 测试多模态智能 Agent")
    print("=" * 70)
    
    # 获取 Agent 实例
    print("\n1. 初始化 Agent...")
    agent = get_agent()
    print(f"   Agent 模式: {'LangChain' if agent.llm else '规则引擎'}")
    
    # 测试加载监测历史
    print("\n2. 测试加载监测历史数据...")
    history_data = agent.load_monitor_history(days=7)
    print(f"   加载到 {len(history_data)} 条历史数据")
    
    # 测试分析趋势
    print("\n3. 测试分析监测趋势...")
    trends = agent.analyze_monitor_trends(history_data)
    print(f"   趋势分析结果: {trends}")
    
    # 测试设置情绪数据
    print("\n4. 测试设置情绪数据...")
    test_emotion_data = {
        'emotion': 'happy',
        'emotion_zh': '快乐',
        'confidence': 0.85,
        'probabilities': {
            'happy': 0.85,
            'neutral': 0.10,
            'surprised': 0.05
        },
        'timestamp': datetime.now().isoformat()
    }
    agent.set_emotion_data(test_emotion_data)
    print("   情绪数据设置成功")
    
    # 测试对话
    print("\n5. 测试与 Agent 对话...")
    print("\n用户: 你好，我今天感觉怎么样？")
    result = agent.chat("你好，我今天感觉怎么样？")
    print(f"\nAgent: {result['response']}")
    print(f"   响应模式: {result['mode']}")
    print(f"   耗时: {result['elapsed_seconds']:.2f}秒")
    
    # 测试状态
    print("\n6. 获取 Agent 状态...")
    summary = agent.get_conversation_summary()
    print(f"   对话总数: {summary['total_messages']}")
    print(f"   当前情绪: {summary['current_emotion']}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    try:
        test_agent()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
