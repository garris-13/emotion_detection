"""
USB摄像头监测器 - 修复版本
"""

import sys
import os
import time
import threading
import json
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查 OpenCV
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV 不可用，摄像头功能将被禁用")

# 尝试导入模型
try:
    from models.emotion_model import load_model, EmotionRecognitionModel

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  模型模块不可用，将使用模拟数据")


class USBCameraMonitor:
    """USB摄像头监测器"""

    def __init__(self, model_path=None, capture_interval=5, save_dir="data/monitor_results"):
        self.capture_interval = capture_interval
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)

        # 监测状态
        self.is_monitoring = False
        self.is_paused = False
        self.total_captures = 0
        self.successful_analyses = 0
        self.camera = None
        self.monitor_thread = None

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 监测器设备: {self.device}")

        # 尝试加载模型
        self.model = None
        if model_path and os.path.exists(model_path) and MODEL_AVAILABLE:
            try:
                self.model = load_model(model_path, model_name='resnet18', num_classes=7, device=self.device)
                print("✅ 监测模型加载成功")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 情绪映射
        self.emotion_zh = {
            'anger': '愤怒',
            'disgust': '厌恶',
            'fear': '恐惧',
            'happy': '快乐',
            'sad': '悲伤',
            'surprised': '惊讶',
            'neutral': '平静'
        }

        print(f"📂 监测器初始化完成，保存目录: {os.path.abspath(save_dir)}")

    def check_camera(self, camera_index=0):
        """检查摄像头是否可用"""
        if not CV2_AVAILABLE:
            return False, "OpenCV 未安装"

        try:
            # 尝试不同的后端
            cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)  # Windows

            if not cap.isOpened():
                return False, f"摄像头 {camera_index} 无法打开"

            # 测试读取一帧
            ret, frame = cap.read()
            cap.release()

            if ret:
                return True, f"摄像头 {camera_index} 可用"
            else:
                return False, f"摄像头 {camera_index} 无法读取图像"

        except Exception as e:
            return False, f"摄像头检查异常: {str(e)}"

    def start_monitoring(self, camera_index=0):
        """开始监测"""
        if not CV2_AVAILABLE:
            print("❌ OpenCV 不可用，无法启动摄像头")
            return False

        if self.is_monitoring:
            print("⚠️  监测已在运行中")
            return True

        # 检查摄像头
        success, message = self.check_camera(camera_index)
        if not success:
            print(f"❌ {message}")
            return False

        try:
            print(f"🚀 正在打开摄像头 {camera_index}...")

            # 打开摄像头
            self.camera = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                print("❌ 无法打开摄像头")
                return False

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)

            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ 摄像头已打开: {width}x{height}")

            self.is_monitoring = True
            self.is_paused = False

            # 启动监测线程
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(camera_index,),
                daemon=True
            )
            self.monitor_thread.start()

            print(f"🎬 监测已启动，间隔: {self.capture_interval}秒")
            return True

        except Exception as e:
            print(f"❌ 启动监测失败: {e}")
            if self.camera:
                self.camera.release()
                self.camera = None
            return False

    def _monitoring_loop(self, camera_index):
        """监测循环"""
        print(f"🔍 监测循环开始 (摄像头: {camera_index})")

        while self.is_monitoring:
            try:
                if self.is_paused:
                    time.sleep(0.5)
                    continue

                # 读取摄像头帧
                ret, frame = self.camera.read()
                if not ret:
                    print("⚠️  读取摄像头失败")
                    time.sleep(1)
                    continue

                # 保存图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_filename = f"capture_{timestamp}.jpg"
                image_path = os.path.join(self.save_dir, "images", image_filename)

                # 保存图像文件（使用 imencode + 手动写文件，避免 OpenCV 不支持中文路径的问题）
                try:
                    ret, buf = cv2.imencode('.jpg', frame)
                    if ret:
                        with open(image_path, 'wb') as f:
                            f.write(buf.tobytes())
                    success = ret
                    if success:
                        self.total_captures += 1
                        print(f"📸 保存第 {self.total_captures} 张图片")

                        # 分析图像
                        result = self._analyze_image(frame, timestamp, image_filename)

                        if result:
                            self.successful_analyses += 1
                            self._save_result(result, timestamp, image_filename)

                            emotion = result.get('emotion_zh', '未知')
                            confidence = result.get('confidence', 0)
                            print(f"✅ 分析完成: {emotion} ({confidence:.1%})")

                    else:
                        print(f"❌ 保存图片失败: {image_path}")

                except Exception as e:
                    print(f"❌ 保存图片异常: {e}")

                # 等待指定的间隔时间
                time.sleep(self.capture_interval)

            except Exception as e:
                print(f"❌ 监测循环异常: {e}")
                time.sleep(1)

    def _analyze_image(self, frame, timestamp, image_filename):
        """分析图像"""
        try:
            if self.model is None:
                # 如果没有模型，生成模拟数据
                import random
                emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised']
                main_emotion = random.choice(emotions)
                confidence = random.uniform(0.6, 0.95)

                probabilities = {}
                for emotion in emotions:
                    if emotion == main_emotion:
                        probabilities[emotion] = confidence
                    else:
                        probabilities[emotion] = (1 - confidence) / (len(emotions) - 1)
            else:
                # 使用真实模型分析
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # 预处理
                img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

                # 推理
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probabilities_tensor = torch.softmax(outputs, dim=1)[0]
                    predicted_idx = torch.argmax(probabilities_tensor).item()
                    confidence = probabilities_tensor[predicted_idx].item()

                emotions_list = list(self.emotion_zh.keys())
                main_emotion = emotions_list[predicted_idx]

                # 构建概率字典
                probabilities = {}
                for i, emotion in enumerate(emotions_list):
                    probabilities[emotion] = float(probabilities_tensor[i])

            # 构建结果
            result = {
                'timestamp': datetime.now().isoformat(),
                'emotion': main_emotion,
                'emotion_zh': self.emotion_zh.get(main_emotion, main_emotion),
                'confidence': float(confidence),
                'probabilities': probabilities,
                'image_filename': image_filename,
                'image_path': f"images/{image_filename}"
            }

            return result

        except Exception as e:
            print(f"❌ 图像分析失败: {e}")
            return None

    def _save_result(self, result, timestamp, image_filename):
        """保存结果"""
        try:
            result_filename = f"result_{timestamp}.json"
            result_path = os.path.join(self.save_dir, "results", result_filename)

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"💾 结果已保存: {result_filename}")

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

    def pause_monitoring(self):
        """暂停监测"""
        if self.is_monitoring and not self.is_paused:
            self.is_paused = True
            print("⏸️ 监测已暂停")
            return True
        return False

    def resume_monitoring(self):
        """继续监测"""
        if self.is_monitoring and self.is_paused:
            self.is_paused = False
            print("▶️ 监测已继续")
            return True
        return False

    def stop_monitoring(self):
        """停止监测"""
        if self.is_monitoring:
            print("🛑 正在停止监测...")
            self.is_monitoring = False
            self.is_paused = False

            # 等待线程结束
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
                print("✅ 监测线程已停止")

            # 关闭摄像头
            if self.camera:
                self.camera.release()
                self.camera = None
                print("✅ 摄像头已释放")

            print(f"📊 统计: 共抓拍 {self.total_captures} 张，成功分析 {self.successful_analyses} 张")
            return True
        return False

    def get_status(self):
        """获取状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'is_paused': self.is_paused,
            'total_captures': self.total_captures,
            'successful_analyses': self.successful_analyses,
            'capture_interval': self.capture_interval,
            'save_dir': os.path.abspath(self.save_dir),
            'camera_available': CV2_AVAILABLE,
            'camera_opened': self.camera is not None and hasattr(self.camera, 'isOpened') and self.camera.isOpened(),
            'model_loaded': self.model is not None
        }

    def analyze_history(self, days=None):
        """分析历史数据"""
        results_dir = os.path.join(self.save_dir, "results")

        if not os.path.exists(results_dir):
            return {
                'success': False,
                'error': '没有历史数据',
                'total_results': 0
            }

        try:
            # 读取所有结果文件
            all_results = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            all_results.append(result)
                    except:
                        continue

            if not all_results:
                return {
                    'success': False,
                    'error': '没有分析结果',
                    'total_results': 0
                }

            # 简单分析
            emotion_counts = {}
            for result in all_results:
                emotion = result.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            return {
                'success': True,
                'total_results': len(all_results),
                'analysis': {
                    'emotion_distribution': emotion_counts,
                    'total_samples': len(all_results)
                },
                'summary': f"共分析 {len(all_results)} 条历史数据"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_results': 0
            }


# 全局监测器实例
usb_monitor_instance = None


def get_usb_monitor():
    """获取USB摄像头监测器实例"""
    global usb_monitor_instance

    if usb_monitor_instance is None:
        print("🔄 创建USB摄像头监测器...")

        # 设置保存目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(base_dir, "data", "monitor_results")

        # 查找模型文件
        model_path = None
        possible_paths = [
            os.path.join(base_dir, 'best_model.pth'),
            os.path.join(base_dir, 'models', 'best_model.pth')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ 找到模型文件: {path}")
                break

        usb_monitor_instance = USBCameraMonitor(
            model_path=model_path,
            save_dir=save_dir
        )

        print("✅ USB摄像头监测器创建完成")

    return usb_monitor_instance


if __name__ == "__main__":
    print("USB摄像头监测器测试")
    monitor = get_usb_monitor()
    print(monitor.get_status())