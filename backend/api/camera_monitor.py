"""
摄像头实时监测模块 - USB摄像头完整修复版
用于控制USB摄像头拍照和保存结果
"""

import sys
import os
import time
import threading
import json
from datetime import datetime
import traceback
import re

# 添加后端目录和项目根目录到路径
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_ROOT)
sys.path.append(BACKEND_ROOT)
sys.path.append(PROJECT_ROOT)

# ================ 检查 OpenCV ================
try:
    import cv2

    CV2_AVAILABLE = True
    print(f"✅ OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    CV2_AVAILABLE = False
    print(f"❌ 无法导入 OpenCV: {e}")
    print("请运行: pip install opencv-python")

# ================ 检查模型 ================
try:
    from models.emotion_model import load_model, EmotionRecognitionModel

    MODEL_AVAILABLE = True
    print("✅ 成功导入表情识别模型")
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"❌ 导入表情识别模型失败: {e}")


    # 创建虚拟模型类
    class EmotionRecognitionModel:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            pass

        def to(self, device):
            return self


    def load_model(*args, **kwargs):
        return EmotionRecognitionModel()

# ================ 导入其他模块 ================
from PIL import Image

# 在某些 Windows 环境中, PyTorch 可能因缺少 DLL 导致导入失败 (WinError 1114)
# 在模块级别防护导入，确保在不可用时降级并允许模块被导入。
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception as e:
    torch = None
    transforms = None
    TORCH_AVAILABLE = False
    print(f"⚠️ camera_monitor: PyTorch 导入失败，摄像头监测相关功能将降级: {e}")

# Facenet 人脸检测（可选）
try:
    from facenet import MTCNN
    FACENET_AVAILABLE = True
except Exception as e:
    MTCNN = None
    FACENET_AVAILABLE = False
    print(f"⚠️ facenet MTCNN 导入失败，将回退到 OpenCV 人脸检测: {e}")


MONITOR_SCOPE_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def sanitize_monitor_scope_id(scope_id):
    """登录会话抓拍目录用的 scope，仅接受标准 UUID 字符串。"""
    if not scope_id or not isinstance(scope_id, str):
        return None
    s = scope_id.strip()
    return s.lower() if MONITOR_SCOPE_UUID_RE.match(s) else None


class CameraMonitor:
    """USB摄像头监测器"""

    def __init__(self, model_path=None, capture_interval=5, save_dir="data/monitor_results", user_id=1, session_id=None, monitor_scope_id=None, db_manager=None):
        """
        初始化摄像头监测器

        Args:
            model_path: 模型文件路径
            capture_interval: 抓拍间隔（秒）
            save_dir: 保存目录
            user_id: 用户ID
            session_id: 会话ID
            monitor_scope_id: 本次登录抓拍作用域（UUID），与 user_id 组合为独立子目录
            db_manager: 数据库管理器
        """
        self.capture_interval = capture_interval
        self.save_dir = save_dir
        self.user_id = user_id
        self.session_id = session_id
        self.monitor_scope_id = monitor_scope_id
        self.db_manager = db_manager

        # 检查OpenCV可用性
        if not CV2_AVAILABLE:
            print("⚠️  OpenCV未安装，摄像头功能不可用")
            print("请运行: pip install opencv-python")
            self.camera_available = False
        else:
            self.camera_available = True

        # 监测状态
        self.is_monitoring = False
        self.is_paused = False
        self.total_captures = 0
        self.successful_analyses = 0
        self.camera = None
        self.monitor_thread = None
        self._preview_lock = threading.Lock()
        self._preview_jpeg = None
        self._last_capture_time = 0.0
        self._latest_face_bbox = None
        self._latest_frame_size = None
        self._last_face_detect_time = 0.0
        self._face_detect_interval = 0.35

        os.makedirs(save_dir, exist_ok=True)
        self.images_dir = None
        self.results_dir = None
        self._refresh_storage_paths()

        # 设备设置（只有在 torch 可用时才设置）
        if TORCH_AVAILABLE and torch is not None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"📱 监测器使用设备: {self.device}")
        else:
            self.device = None
            print("⚠️ PyTorch 不可用，监测器将使用模拟分析（如果模型不可用）")

        # 尝试加载模型（仅在模型模块和 torch 可用时）
        self.model = None
        if TORCH_AVAILABLE and MODEL_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path, model_name='resnet18', num_classes=7, device=self.device)
                print("✅ 监测模型加载成功")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None

        # 图像预处理（仅在 torchvision.transforms 可用时初始化）
        if TORCH_AVAILABLE and transforms is not None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = None

        # 初始化人脸检测器
        self.face_detector = None
        self.face_cascade = None
        self.face_detection_method = 'none'

        if FACENET_AVAILABLE and TORCH_AVAILABLE and MTCNN is not None:
            try:
                detector_device = self.device if self.device is not None else 'cpu'
                self.face_detector = MTCNN(keep_all=True, device=detector_device)
                self.face_detection_method = 'facenet_mtcnn'
                print(f"✅ 人脸检测器已启用: {self.face_detection_method}")
            except Exception as e:
                print(f"⚠️ 初始化 facenet MTCNN 失败: {e}")

        if self.face_detector is None and CV2_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                if self.face_cascade is not None and not self.face_cascade.empty():
                    self.face_detection_method = 'opencv_haar'
                    print(f"✅ 人脸检测器已启用: {self.face_detection_method}")
                else:
                    self.face_cascade = None
            except Exception as e:
                print(f"⚠️ 初始化 OpenCV Haar 人脸检测失败: {e}")

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

        # 情绪标签列表
        self.emotions = list(self.emotion_zh.keys())

        print(f"📂 监测数据保存到: {os.path.abspath(save_dir)}")
        print(f"📂 当前抓拍目录: {os.path.abspath(self.images_dir)}")

    def _refresh_storage_paths(self):
        scope = sanitize_monitor_scope_id(self.monitor_scope_id)
        if scope:
            self.images_dir = os.path.join(self.save_dir, "images", str(self.user_id), scope)
            self.results_dir = os.path.join(self.save_dir, "results", str(self.user_id), scope)
        else:
            self.images_dir = os.path.join(self.save_dir, "images")
            self.results_dir = os.path.join(self.save_dir, "results")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def set_scope(self, user_id, session_id=None, monitor_scope_id=None, db_manager=None):
        self.user_id = user_id
        if session_id is not None:
            self.session_id = session_id
        if db_manager is not None:
            self.db_manager = db_manager
        if monitor_scope_id is not None and str(monitor_scope_id).strip() != '':
            sanitized = sanitize_monitor_scope_id(monitor_scope_id)
            if sanitized:
                self.monitor_scope_id = sanitized
        self._refresh_storage_paths()

    def check_camera(self, camera_index=0):
        """
        检查摄像头连接

        Args:
            camera_index: 摄像头索引

        Returns:
            tuple: (success, message)
        """
        if not self.camera_available:
            return False, "OpenCV未安装"

        try:
            print(f"检查摄像头 {camera_index}...")

            # 尝试用DirectShow后端（Windows最稳定）
            cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)

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
            return False, f"摄像头检查失败: {str(e)}"

    def start(self, camera_index=0, capture_interval=5):
        """
        启动监测（兼容 API 接口）

        Args:
            camera_index: 摄像头索引
            capture_interval: 抓拍间隔（秒）

        Returns:
            dict: 启动结果
        """
        self.capture_interval = capture_interval
        success = self.start_monitoring(camera_index)
        if success:
            return {
                "status": "started",
                "message": f"摄像头监测已启动 (摄像头索引: {camera_index}, 抓拍间隔: {capture_interval}秒)"
            }
        else:
            return {
                "status": "error",
                "message": "摄像头监测启动失败"
            }

    def start_monitoring(self, camera_index=0):
        """
        开始监测

        Args:
            camera_index: 摄像头索引

        Returns:
            bool: 是否成功启动
        """
        if not self.camera_available:
            print("❌ OpenCV未安装，无法使用摄像头")
            return False

        if self.is_monitoring:
            print("⚠️  监测已在运行中")
            return True

        try:
            # 检查摄像头连接
            success, message = self.check_camera(camera_index)
            if not success:
                print(f"❌ {message}")
                return False

            print(f"🚀 正在打开摄像头 {camera_index}...")

            # 尝试用DirectShow后端（Windows）
            self.camera = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                print(f"❌ 无法打开摄像头 {camera_index}")
                return False

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)

            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)

            print(f"✅ 摄像头 {camera_index} 已打开")
            print(f"📊 分辨率: {width}x{height}, FPS: {fps:.1f}")
            print(f"📸 抓拍间隔: {self.capture_interval}秒")

            self.is_monitoring = True
            self.is_paused = False
            self._last_capture_time = 0.0

            # 启动监测线程
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(camera_index,),
                daemon=True
            )
            self.monitor_thread.start()

            print("🎬 监测线程已启动")
            return True

        except Exception as e:
            print(f"❌ 启动监测失败: {e}")
            traceback.print_exc()
            if self.camera:
                self.camera.release()
                self.camera = None
            return False

    def _set_preview_from_frame(self, frame):
        """将当前帧编码为 JPEG 供前端 MJPEG 预览（在监测线程内调用）"""
        if not CV2_AVAILABLE or frame is None:
            return
        try:
            h, w = frame.shape[:2]
            scale = min(1.0, 640.0 / max(w, 1))
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                small = frame
            ok, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 68])
            if ok:
                with self._preview_lock:
                    self._preview_jpeg = buf.tobytes()
        except Exception:
            pass

    def get_preview_jpeg(self):
        """返回最近一次预览 JPEG 字节，若无则 None"""
        with self._preview_lock:
            return self._preview_jpeg

    def _monitoring_loop(self, camera_index):
        """监测循环（持续刷新预览；按间隔抓拍与分析）"""
        print(f"🔍 监测循环开始 (摄像头: {camera_index}, 间隔: {self.capture_interval}s)")

        while self.is_monitoring:
            try:
                if self.is_paused:
                    ret, frame = self.camera.read()
                    if ret:
                        self._set_preview_from_frame(frame)
                    time.sleep(0.12)
                    continue

                ret, frame = self.camera.read()
                if not ret:
                    print("⚠️  读取摄像头帧失败")
                    time.sleep(0.25)
                    continue

                now = time.time()
                if (now - self._last_face_detect_time) >= self._face_detect_interval:
                    latest_bbox = self._detect_face_bbox(frame)
                    self._latest_face_bbox = latest_bbox
                    self._latest_frame_size = {
                        'width': int(frame.shape[1]),
                        'height': int(frame.shape[0])
                    }
                    self._last_face_detect_time = now

                self._set_preview_from_frame(frame)

                elapsed = now - self._last_capture_time
                if elapsed < self.capture_interval:
                    time.sleep(min(0.05, max(0.02, self.capture_interval - elapsed)))
                    continue

                self._last_capture_time = now

                # 保存图像文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_filename = f"capture_{timestamp}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)

                # 保存图像
                try:
                    # 使用 imencode + 手动写文件，避免 OpenCV 不支持中文路径的问题
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    ret, buf = cv2.imencode('.jpg', frame, encode_param)
                    if ret:
                        with open(image_path, 'wb') as f:
                            f.write(buf.tobytes())
                    success = ret

                    if success:
                        self.total_captures += 1
                        print(f"📸 保存第 {self.total_captures} 张图片: {image_filename}")

                        # 分析图像
                        result = self._analyze_frame(frame, timestamp, image_filename)

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

            except Exception as e:
                print(f"❌ 监测循环错误: {e}")
                traceback.print_exc()
                time.sleep(1)

    def _analyze_frame(self, frame, timestamp, image_filename):
        """
        分析摄像头帧

        Args:
            frame: OpenCV图像帧
            timestamp: 时间戳
            image_filename: 图像文件名

        Returns:
            dict: 分析结果或None
        """
        face_frame, face_bbox = self._detect_face_region(frame)

        if face_bbox:
            print(
                f"🧩 检测到人脸框: x={face_bbox['x']}, y={face_bbox['y']}, "
                f"w={face_bbox['width']}, h={face_bbox['height']}, "
                f"score={face_bbox['confidence']:.3f}"
            )
        else:
            print("⚠️  未检测到人脸，回退使用全图")

        if self.model is None:
            # 没有模型，生成模拟数据
            print("⚠️  使用模拟分析结果")
            result = self._simulate_analysis(face_frame, timestamp, image_filename)
            if result is not None:
                result['face_detected'] = face_bbox is not None
                result['face_bbox'] = face_bbox
                result['face_detection_method'] = self.face_detection_method
            return result

        try:
            # 转换OpenCV BGR到RGB
            rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 显示图像信息
            print(f"🔍 分析图像: {pil_image.size}像素, 模式: {pil_image.mode}")

            # 预处理
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

            emotion = self.emotions[predicted_idx]

            # 构造概率字典
            prob_dict = {}
            for i, emotion_name in enumerate(self.emotions):
                prob_dict[emotion_name] = float(probabilities[i])

            result = {
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion,
                'emotion_zh': self.emotion_zh.get(emotion, emotion),
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'image_filename': image_filename,
                'image_path': f"images/{image_filename}",
                'face_detected': face_bbox is not None,
                'face_bbox': face_bbox,
                'face_detection_method': self.face_detection_method
            }

            return result

        except Exception as e:
            print(f"❌ 分析帧失败: {e}")
            traceback.print_exc()
            return None

    def _simulate_analysis(self, frame, timestamp, image_filename):
        """模拟情绪分析（用于测试）"""
        try:
            import random

            # 模拟随机情绪
            emotions = self.emotions
            main_emotion = random.choice(emotions)
            confidence = random.uniform(0.6, 0.9)

            # 生成模拟的概率分布
            probabilities = {}
            for emotion in emotions:
                if emotion == main_emotion:
                    probabilities[emotion] = confidence
                else:
                    probabilities[emotion] = (1 - confidence) / (len(emotions) - 1)

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
            print(f"❌ 模拟分析失败: {e}")
            return None

    def _save_result(self, result, timestamp, image_filename):
        """
        保存分析结果

        Args:
            result: 分析结果字典
            timestamp: 时间戳
            image_filename: 图像文件名
        """
        try:
            # 添加图像文件信息
            result['image_filename'] = image_filename
            result['image_path'] = f"images/{image_filename}"

            # 保存结果到JSON文件
            result_filename = f"result_{timestamp}.json"
            result_path = os.path.join(self.results_dir, result_filename)

            scope = sanitize_monitor_scope_id(self.monitor_scope_id)
            if scope:
                result['monitor_user_id'] = self.user_id
                result['monitor_scope_id'] = scope

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"💾 结果已保存: {result_path}")

            # 保存到数据库
            if self.db_manager is not None:
                try:
                    emotion_data = {
                        'emotion': result.get('emotion'),
                        'emotion_zh': result.get('emotion_zh'),
                        'confidence': result.get('confidence'),
                        'probabilities': result.get('probabilities', {})
                    }
                    record_id = self.db_manager.save_emotion_record(
                        user_id=self.user_id,
                        emotion_data=emotion_data,
                        session_id=self.session_id
                    )
                    print(f"✅ 情绪记录已保存到数据库，记录ID: {record_id}")
                except Exception as db_error:
                    print(f"⚠️ 保存到数据库失败: {db_error}")

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

    def pause(self):
        """暂停监测（兼容 API 接口）"""
        if self.is_monitoring and not self.is_paused:
            self.is_paused = True
            print("⏸️ 监测已暂停")
            return {"status": "paused"}
        return {"status": "error", "message": "无法暂停监测"}

    def pause_monitoring(self):
        """暂停监测"""
        return self.pause()

    def resume(self):
        """继续监测（兼容 API 接口）"""
        if self.is_monitoring and self.is_paused:
            self.is_paused = False
            print("▶️ 监测已继续")
            return {"status": "resumed"}
        return {"status": "error", "message": "无法继续监测"}

    def resume_monitoring(self):
        """继续监测"""
        return self.resume()

    def stop(self):
        """停止监测（兼容 API 接口）"""
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

            with self._preview_lock:
                self._preview_jpeg = None
            self._latest_face_bbox = None
            self._latest_frame_size = None
            self._last_face_detect_time = 0.0

            print(f"📊 统计: 共抓拍 {self.total_captures} 张，成功分析 {self.successful_analyses} 张")
            return {"status": "stopped"}
        return {"status": "error", "message": "监测未运行"}

    def stop_monitoring(self):
        """停止监测"""
        return self.stop()

    def get_status(self):
        """
        获取监测状态

        Returns:
            dict: 状态信息
        """
        return {
            'is_monitoring': self.is_monitoring,
            'is_paused': self.is_paused,
            'total_captures': self.total_captures,
            'successful_analyses': self.successful_analyses,
            'capture_interval': self.capture_interval,
            'save_dir': os.path.abspath(self.save_dir),
            'model_loaded': self.model is not None,
            'face_detection_method': self.face_detection_method,
            'latest_face_bbox': self._latest_face_bbox,
            'latest_frame_size': self._latest_frame_size,
            'camera_available': self.camera_available,
            'camera_opened': self.camera is not None and hasattr(self.camera, 'isOpened') and self.camera.isOpened()
        }

    def _detect_face_bbox(self, frame):
        """
        检测人脸并返回边框

        Returns:
            dict or None: {'x','y','width','height','confidence'}
        """
        frame_h, frame_w = frame.shape[:2]

        # 优先使用 facenet MTCNN
        if self.face_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = self.face_detector.detect(rgb_frame)

                if boxes is not None and len(boxes) > 0:
                    valid_boxes = []
                    for idx, box in enumerate(boxes):
                        if box is None:
                            continue

                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1 = max(0, min(frame_w - 1, x1))
                        y1 = max(0, min(frame_h - 1, y1))
                        x2 = max(0, min(frame_w, x2))
                        y2 = max(0, min(frame_h, y2))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        score = 0.0
                        if probs is not None and idx < len(probs) and probs[idx] is not None:
                            score = float(probs[idx])

                        width = x2 - x1
                        height = y2 - y1
                        area = width * height

                        valid_boxes.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'width': width,
                            'height': height,
                            'area': area,
                            'score': score
                        })

                    if valid_boxes:
                        best_face = max(valid_boxes, key=lambda b: (b['score'], b['area']))
                        pad = int(max(best_face['width'], best_face['height']) * 0.12)

                        x1 = max(0, best_face['x1'] - pad)
                        y1 = max(0, best_face['y1'] - pad)
                        x2 = min(frame_w, best_face['x2'] + pad)
                        y2 = min(frame_h, best_face['y2'] + pad)

                        return {
                            'x': int(x1),
                            'y': int(y1),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'confidence': float(best_face['score'])
                        }
            except Exception as e:
                print(f"⚠️ facenet 人脸检测异常，回退到 OpenCV: {e}")

        # 回退到 OpenCV Haar
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    pad = int(max(w, h) * 0.12)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame_w, x + w + pad)
                    y2 = min(frame_h, y + h + pad)

                    return {
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'confidence': 1.0
                    }
            except Exception as e:
                print(f"⚠️ OpenCV 人脸检测异常: {e}")

        return None

    def _detect_face_region(self, frame):
        """
        检测人脸并返回用于后续处理的人脸区域

        Returns:
            tuple: (face_frame, face_bbox)
        """
        frame_h, frame_w = frame.shape[:2]
        face_bbox = self._detect_face_bbox(frame)
        if face_bbox is None:
            return frame, None

        x1 = max(0, int(face_bbox['x']))
        y1 = max(0, int(face_bbox['y']))
        x2 = min(frame_w, x1 + int(face_bbox['width']))
        y2 = min(frame_h, y1 + int(face_bbox['height']))

        if x2 <= x1 or y2 <= y1:
            return frame, None

        face_frame = frame[y1:y2, x1:x2]
        return face_frame, face_bbox

    def analyze_history(self, days=None):
        """分析历史数据"""
        results_dir = self.results_dir

        if not os.path.exists(results_dir):
            return {
                'success': False,
                'error': '没有找到历史数据目录',
                'total_results': 0,
                'results_dir': results_dir
            }

        try:
            # 收集所有结果文件
            result_files = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    result_files.append(filepath)

            if not result_files:
                return {
                    'success': False,
                    'error': '没有分析结果文件',
                    'total_results': 0
                }

            print(f"📊 分析 {len(result_files)} 个结果文件...")

            # 读取结果
            all_results = []
            for filepath in result_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        all_results.append(result)
                except Exception as e:
                    print(f"❌ 读取文件失败 {filepath}: {e}")
                    continue

            if not all_results:
                return {
                    'success': False,
                    'error': '无法读取结果文件',
                    'total_results': 0
                }

            # 按时间筛选
            if days is not None:
                cutoff_time = time.time() - (days * 24 * 3600)
                filtered_results = []
                for result in all_results:
                    try:
                        result_time = datetime.fromisoformat(result['timestamp']).timestamp()
                        if result_time >= cutoff_time:
                            filtered_results.append(result)
                    except:
                        continue
                all_results = filtered_results

            # 进行综合分析
            analysis = self._comprehensive_analysis(all_results)

            # 生成健康建议
            health_advice = self._generate_health_advice(analysis)

            return {
                'success': True,
                'total_results': len(all_results),
                'analysis': analysis,
                'summary': self._generate_summary(all_results, analysis, health_advice)  # 传入所有参数
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_results': 0
            }

    def _comprehensive_analysis(self, results):
        """综合分析"""
        if not results:
            return {}

        # 统计情绪频率
        emotion_counts = {}
        emotion_confidences = {}

        for result in results:
            emotion = result.get('emotion', 'unknown')
            confidence = result.get('confidence', 0)

            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_confidences[emotion] = []

            emotion_counts[emotion] += 1
            emotion_confidences[emotion].append(confidence)

        # 计算平均置信度
        avg_confidences = {}
        for emotion, conf_list in emotion_confidences.items():
            if conf_list:
                avg_confidences[emotion] = sum(conf_list) / len(conf_list)
            else:
                avg_confidences[emotion] = 0

        # 找到主要情绪
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        else:
            dominant_emotion = ('unknown', 0)

        return {
            'emotion_distribution': emotion_counts,
            'average_confidences': avg_confidences,
            'dominant_emotion': {
                'emotion': dominant_emotion[0],
                'emotion_zh': self.emotion_zh.get(dominant_emotion[0], dominant_emotion[0]),
                'count': dominant_emotion[1],
                'percentage': (dominant_emotion[1] / len(results)) * 100 if results else 0
            },
            'total_samples': len(results)
        }

    def analyze_history_with_advice(self, days=None):
        """分析历史数据并生成健康建议"""
        results_dir = self.results_dir

        if not os.path.exists(results_dir):
            return {
                'success': False,
                'error': '没有历史数据目录',
                'total_results': 0
            }

        try:
            # 收集所有结果文件
            result_files = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    result_files.append(filepath)

            if not result_files:
                return {
                    'success': False,
                    'error': '没有分析结果',
                    'total_results': 0
                }

            print(f"📊 分析 {len(result_files)} 个结果文件...")

            # 读取结果
            all_results = []
            for filepath in result_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        all_results.append(result)
                except Exception as e:
                    print(f"❌ 读取文件失败 {filepath}: {e}")
                    continue

            if not all_results:
                return {
                    'success': False,
                    'error': '无法读取结果文件',
                    'total_results': 0
                }

            # 综合分析
            analysis = self._comprehensive_analysis(all_results)

            # 生成健康建议
            health_advice = self._generate_health_advice(analysis)

            # 生成总结报告
            summary = self._generate_summary(all_results, analysis, health_advice)

            return {
                'success': True,
                'total_results': len(all_results),
                'analysis': analysis,
                'health_advice': health_advice,
                'summary': summary,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_results': 0
            }

    def _generate_health_advice(self, analysis):
        """生成健康建议"""
        try:
            # 尝试导入健康建议模块
            try:
                from models.health_advisor import HealthAdvisor, EmotionResult, create_advice_from_probabilities
                advisor_available = True
            except ImportError:
                advisor_available = False
                print("⚠️  健康建议模块不可用")

            if not advisor_available:
                # 返回简单建议
                dominant_emotion = analysis.get('dominant_emotion', {})
                emotion = dominant_emotion.get('emotion', 'unknown')
                emotion_zh = self.emotion_zh.get(emotion, emotion)

                return {
                    'description': f'基于历史数据分析，您的主要情绪是{emotion_zh}',
                    'recommendations': [
                        '建议定期进行情绪记录',
                        '注意情绪变化趋势',
                        '保持健康的生活方式'
                    ],
                    'risk_level': 'low' if emotion in ['happy', 'surprised'] else 'medium'
                }

            # 使用健康建议模块
            emotion_distribution = analysis.get('emotion_distribution', {})
            total_samples = analysis.get('total_samples', 1)

            # 计算平均概率
            probabilities = {}
            for emotion, count in emotion_distribution.items():
                probabilities[emotion] = count / total_samples

            # 确保所有情绪都有概率
            for emotion in self.emotions:
                if emotion not in probabilities:
                    probabilities[emotion] = 0.0

            # 生成建议
            report = create_advice_from_probabilities(probabilities)

            # 提取建议信息
            health_advice = {
                'description': report['health_advice']['description'] if 'health_advice' in report else '情绪健康建议',
                'immediate_actions': report['health_advice'].get('immediate_actions', []),
                'daily_tips': report['health_advice'].get('daily_tips', []),
                'long_term_suggestions': report['health_advice'].get('long_term_suggestions', []),
                'risk_level': report['risk_assessment'].get('risk_level',
                                                            'unknown') if 'risk_assessment' in report else 'unknown'
            }

            return health_advice

        except Exception as e:
            print(f"❌ 生成健康建议失败: {e}")
            return {
                'description': '情绪分析报告',
                'recommendations': ['保持积极心态', '注意情绪管理'],
                'risk_level': 'unknown'
            }

    def _generate_summary(self, all_results, analysis=None, health_advice=None):
        """生成详细总结报告"""

        # 如果没有传入analysis，则进行计算
        if analysis is None:
            analysis = self._comprehensive_analysis(all_results)

        # 如果没有传入health_advice，则生成
        if health_advice is None:
            health_advice = self._generate_health_advice(analysis)

        summary = f"📊 综合情绪分析报告\n"
        summary += f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"📈 分析样本: {len(all_results)} 条数据\n\n"

        # 情绪分布
        summary += "🎭 情绪分布:\n"
        emotion_distribution = analysis.get('emotion_distribution', {})
        for emotion, count in sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_results)) * 100 if all_results else 0
            emotion_name = self.emotion_zh.get(emotion, emotion)
            summary += f"  {emotion_name}: {count}次 ({percentage:.1f}%)\n"

        # 主要情绪
        dominant = analysis.get('dominant_emotion', {})
        if dominant:
            summary += f"\n👑 主要情绪: {dominant.get('emotion_zh', dominant.get('emotion', '未知'))}\n"
            summary += f"   出现次数: {dominant.get('count', 0)}\n"
            summary += f"   占比: {dominant.get('percentage', 0):.1f}%\n"

        # 健康建议
        if health_advice:
            summary += f"\n💡 健康建议:\n"
            summary += f"   {health_advice.get('description', '暂无建议')}\n"

            if 'immediate_actions' in health_advice and health_advice['immediate_actions']:
                summary += "\n   🚨 立即行动:\n"
                for i, action in enumerate(health_advice['immediate_actions'][:3], 1):
                    summary += f"     {i}. {action}\n"

            if 'daily_tips' in health_advice and health_advice['daily_tips']:
                summary += "\n   📅 日常贴士:\n"
                for i, tip in enumerate(health_advice['daily_tips'][:3], 1):
                    summary += f"     {i}. {tip}\n"

            # 风险评估
            risk_level = health_advice.get('risk_level', 'unknown')
            risk_map = {
                'very_low': '🟢 风险极低',
                'low': '🟢 风险低',
                'medium': '🟡 风险中等',
                'high': '🟠 风险较高',
                'very_high': '🔴 风险很高'
            }
            summary += f"\n⚠️  风险评估: {risk_map.get(risk_level, '未知')}\n"

        return summary

    def get_recent_results(self, limit=10):
        """获取最近的结果"""
        results_dir = self.results_dir

        if not os.path.exists(results_dir):
            return []

        try:
            # 获取所有JSON文件并按时间排序
            result_files = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    mod_time = os.path.getmtime(filepath)
                    result_files.append((mod_time, filepath))

            # 按时间排序
            result_files.sort(reverse=True)

            # 读取最近的结果
            recent_results = []
            for _, filepath in result_files[:limit]:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        recent_results.append(result)
                except:
                    continue

            return recent_results
        except Exception as e:
            print(f"获取最近结果失败: {e}")
            return []


# 全局监测器实例
global_monitor = None


def get_monitor(model_path=None, save_dir=None, user_id=1, session_id=None, monitor_scope_id=None, db_manager=None):
    """获取全局监测器实例"""
    global global_monitor

    if global_monitor is None:
        if save_dir is None:
            # 使用默认保存目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(base_dir, "data", "monitor_results")

        print(f"📁 初始化摄像头监测器...")
        print(f"📁 保存目录: {save_dir}")

        global_monitor = CameraMonitor(
            model_path=model_path,
            save_dir=save_dir,
            user_id=user_id,
            session_id=session_id,
            monitor_scope_id=monitor_scope_id,
            db_manager=db_manager
        )
        print("✅ 摄像头监测器初始化完成")
    else:
        global_monitor.set_scope(
            user_id,
            session_id=session_id,
            monitor_scope_id=monitor_scope_id,
            db_manager=db_manager,
        )

    return global_monitor


# 测试函数
if __name__ == "__main__":
    print("=" * 60)
    print("摄像头监测模块测试")
    print("=" * 60)

    # 创建监测器
    monitor = get_monitor()

    # 显示状态
    status = monitor.get_status()
    print(f"状态: {status}")

    # 测试功能
    print("\n测试功能:")
    print("1. 启动监测 (5秒)")
    print("2. 暂停/继续")
    print("3. 停止监测")
    print("4. 分析历史数据")

    choice = input("\n选择测试 (1-4): ")

    if choice == '1':
        if monitor.start_monitoring():
            print("监测已启动，等待5秒...")
            time.sleep(5)
            monitor.stop_monitoring()
    elif choice == '2':
        print("暂停/继续功能需要先启动监测")
    elif choice == '3':
        monitor.stop_monitoring()
    elif choice == '4':
        analysis = monitor.analyze_history(days=1)
        if analysis['success']:
            print(analysis['summary'])
        else:
            print(f"分析失败: {analysis['error']}")