"""
表情识别 API 使用示例
"""

import sys
import os

# Ensure project root is on sys.path when running from examples/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.api.api_client import EmotionRecognitionClient, print_result
from PIL import Image
import os

def main():
    print("=" * 70)
    print("表情识别 API 使用示例")
    print("=" * 70)
    
    # 创建客户端
    client = EmotionRecognitionClient('http://localhost:5000')
    
    # 1. 检查API健康状态
    print("\n【示例 1】检查 API 健康状态")
    print("-" * 70)
    health = client.check_health()
    print(f"状态: {health.get('status')}")
    print(f"模型已加载: {health.get('model_loaded')}")
    print(f"运行设备: {health.get('device')}")
    
    # 2. 获取支持的表情列表
    print("\n【示例 2】获取支持的表情列表")
    print("-" * 70)
    emotions_info = client.get_emotions()
    print(f"支持 {emotions_info.get('count')} 种表情:")
    for i, emotion in enumerate(emotions_info.get('emotions', []), 1):
        print(f"  {i}. {emotion}")
    
    # 3. 从文件预测（如果存在测试图像）
    test_image = 'test_image.jpg'
    if os.path.exists(test_image):
        print(f"\n【示例 3】从文件预测: {test_image}")
        print("-" * 70)
        result = client.predict_from_file(test_image)
        print_result(result)
    else:
        print(f"\n【示例 3】跳过（测试图像不存在: {test_image}）")
    
    # 4. 从PIL Image预测（创建示例图像）
    print("\n【示例 4】从PIL Image对象预测")
    print("-" * 70)
    try:
        # 创建一个示例图像（实际使用时应该是真实的人脸图像）
        from PIL import Image
        import numpy as np
        
        # 创建一个随机图像（仅作为示例）
        random_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        result = client.predict_from_pil(random_image)
        print("注意: 这是随机图像，预测结果没有实际意义")
        print_result(result)
    except Exception as e:
        print(f"错误: {e}")
    
    # 5. 批量预测示例
    print("\n【示例 5】批量预测")
    print("-" * 70)
    # 这里需要多个测试图像
    test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if len(existing_images) > 0:
        print(f"批量预测 {len(existing_images)} 张图像...")
        batch_result = client.predict_batch(existing_images)
        
        if batch_result.get('success'):
            print(f"✅ 成功预测 {batch_result['count']} 张图像")
            for i, result in enumerate(batch_result['results'], 1):
                print(f"\n图像 {i}:")
                print(f"  表情: {result['emotion']}")
                print(f"  置信度: {result['confidence']:.2%}")
        else:
            print(f"❌ 批量预测失败: {batch_result.get('error')}")
    else:
        print("跳过批量预测（没有找到测试图像）")
    
    print("\n" + "=" * 70)
    print("示例运行完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
