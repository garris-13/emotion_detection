"""
验证表情识别系统完整功能
"""

import sys
import os
import json
import requests
from datetime import datetime

print("=" * 70)
print("🔍 表情识别系统 - 完整功能验证")
print("=" * 70)

# 基础配置
API_URL = "http://localhost:7860"
print(f"📡 API服务器: {API_URL}")


def test_api_connection():
    """测试API连接"""
    print("\n[1/5] 测试API连接...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API连接正常")
            print(f"   状态: {data.get('status', '未知')}")
            print(f"   模型加载: {data.get('model_loaded', '未知')}")
            print(f"   设备: {data.get('device', '未知')}")
            return True
        else:
            print(f"❌ API连接失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API连接异常: {e}")
        return False


def test_monitor_status():
    """测试监测器状态"""
    print("\n[2/5] 测试监测器状态...")
    try:
        response = requests.get(f"{API_URL}/monitor/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                status = data.get('status', {})
                print(f"✅ 监测器状态:")
                print(f"   运行中: {'是' if status.get('is_monitoring') else '否'}")
                print(f"   已暂停: {'是' if status.get('is_paused') else '否'}")
                print(f"   抓拍数: {status.get('total_captures', 0)}")
                print(f"   分析数: {status.get('successful_analyses', 0)}")
                print(f"   摄像头可用: {'是' if status.get('camera_available') else '否'}")
                return True
            else:
                print(f"⚠️  监测器状态获取失败: {data.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ 监测器状态请求失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 监测器状态异常: {e}")
        return False


def test_emotions_list():
    """测试表情列表"""
    print("\n[3/5] 测试表情列表...")
    try:
        response = requests.get(f"{API_URL}/emotions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                emotions = data.get('emotions', [])
                print(f"✅ 支持的表情 ({len(emotions)}种):")
                for emo in emotions:
                    print(f"   {emo.get('en')}: {emo.get('zh')}")
                return True
            else:
                print(f"❌ 表情列表获取失败")
                return False
        else:
            print(f"❌ 表情列表请求失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 表情列表异常: {e}")
        return False


def test_health_advice():
    """测试健康建议"""
    print("\n[4/5] 测试健康建议生成...")
    try:
        # 测试数据
        test_data = {
            "probabilities": {
                "anger": 0.45,
                "disgust": 0.05,
                "fear": 0.10,
                "happy": 0.15,
                "sad": 0.20,
                "surprised": 0.05
            }
        }

        response = requests.post(
            f"{API_URL}/advice/analysis",
            json=test_data,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                report = data.get('report', {})
                print(f"✅ 健康建议生成成功")
                print(f"   主要情绪: {report.get('emotion_analysis', {}).get('main_emotion_zh', '未知')}")
                print(f"   风险等级: {report.get('risk_assessment', {}).get('risk_level', '未知')}")

                # 检查建议内容
                advice = report.get('health_advice', {})
                if advice.get('immediate_actions'):
                    print(f"   立即行动建议: {len(advice['immediate_actions'])}条")
                return True
            else:
                print(f"❌ 健康建议生成失败: {data.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ 健康建议请求失败: HTTP {response.status_code}")
            print(f"   响应: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ 健康建议异常: {e}")
        return False


def test_monitor_analysis():
    """测试监测历史分析"""
    print("\n[5/5] 测试监测历史分析...")
    try:
        response = requests.get(f"{API_URL}/monitor/analyze?days=7", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ 监测历史分析成功")
                print(f"   分析数据量: {data.get('total_results', 0)}条")

                analysis = data.get('analysis', {})
                if analysis:
                    dist = analysis.get('emotion_distribution', {})
                    if dist:
                        print(f"   情绪分布: {len(dist)}种情绪")
                        for emotion, count in dist.items():
                            print(f"     {emotion}: {count}次")

                # 检查健康建议
                health_advice = data.get('health_advice', {})
                if health_advice:
                    print(f"   健康建议: {health_advice.get('description', '无描述')}")
                    print(f"   风险等级: {health_advice.get('risk_level', '未知')}")

                return True
            else:
                print(f"❌ 监测历史分析失败: {data.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ 监测历史分析请求失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 监测历史分析异常: {e}")
        return False


def check_data_directory():
    """检查数据目录"""
    print("\n📁 检查数据目录...")
    try:
        # 检查监测结果目录
        monitor_dir = "data/monitor_results"
        images_dir = os.path.join(monitor_dir, "images")
        results_dir = os.path.join(monitor_dir, "results")

        print(f"   监测目录: {os.path.abspath(monitor_dir)}")

        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
            print(f"   图像文件: {len(images)}个")
            if images:
                for img in images[:3]:  # 显示前3个
                    print(f"     - {img}")

        if os.path.exists(results_dir):
            results = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            print(f"   结果文件: {len(results)}个")
            if results:
                for res in results[:3]:  # 显示前3个
                    print(f"     - {res}")

        return True
    except Exception as e:
        print(f"❌ 检查数据目录异常: {e}")
        return False


def main():
    """主函数"""
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 执行测试
    tests_passed = 0
    total_tests = 5

    if test_api_connection():
        tests_passed += 1

    if test_monitor_status():
        tests_passed += 1

    if test_emotions_list():
        tests_passed += 1

    if test_health_advice():
        tests_passed += 1

    if test_monitor_analysis():
        tests_passed += 1

    check_data_directory()

    print("\n" + "=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    print(f"✅ 通过测试: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 所有测试通过！系统功能正常")
        print("\n🎯 下一步操作:")
        print("   1. 访问 http://127.0.0.1:8000/frontend/examples/emotion_ui.html")
        print("   2. 点击'分析监测历史'查看报告")
        print("   3. 如果需要更多数据，可以启动实时监测")
    else:
        print("⚠️  部分测试失败，需要修复")
        print("\n💡 建议:")
        print("   1. 检查API服务器是否正常运行")
        print("   2. 检查摄像头是否连接")
        print("   3. 查看错误日志进行调试")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
    input("\n按Enter键退出...")