"""
表情识别 REST API - 完整修复版
包含健康建议、摄像头监测和阿里云大模型功能
修复了综合分析功能
"""

import sys
import os

# ================ 修复路径和导入问题 ================

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

print("=" * 70)
print("🚀 表情识别与健康建议 API - AI大模型增强版")
print("=" * 70)
print(f"📁 项目根目录: {PROJECT_ROOT}")

# ================ 检查 OpenCV ================
try:
    import cv2

    CV2_AVAILABLE = True
    print(f"✅ OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    CV2_AVAILABLE = False
    print(f"❌ OpenCV 不可用: {e}")
    print("⚠️  摄像头功能将不可用")

# ================ 继续其他导入 ================
from flask import Flask, request, jsonify
from flask_cors import CORS

# Torch may fail to import on some Windows setups (missing CUDA/VC++ runtime).
# Import defensively so the API can still run in degraded mode without GPU/Torch.
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception as e:
    torch = None
    transforms = None
    TORCH_AVAILABLE = False
    print(f"⚠️ PyTorch 导入失败，进入降级模式: {e}")
from PIL import Image
import io
import base64
import json
from datetime import datetime, timedelta
import time
import numpy as np
try:
    from multiAgent.MultiAgentFlow import run_flow_from_data
    from multiAgent.key_loader import get_api_key

    MULTI_AGENT_AVAILABLE = True
    print("✅ MultiAgent 模块可用")
except Exception as e:
    MULTI_AGENT_AVAILABLE = False
    get_api_key = None
    print(f"⚠️  MultiAgent 模块不可用: {e}")
# MediaPipe 导入可能有 NumPy 兼容性问题，安全导入
MEDIAPIPE_AVAILABLE = False
mp = None
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print(f"✅ MediaPipe 版本: {mp.__version__}")
except Exception as e:
    print(f"⚠️ MediaPipe 导入失败: {e}")
    print("   将使用 OpenCV 进行人脸检测")
# ================ LangGraph Agent 和数据库 ================
langgraph_agent = None
db_manager = None
LANGGRAPH_AVAILABLE = False
DATABASE_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    from langgraph_agent import get_langgraph_agent
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph Agent 模块可用")
except Exception as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️ LangGraph Agent 模块导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from backend.database import get_db_manager
    DATABASE_AVAILABLE = True
    print("✅ 数据库模块可用")
except Exception as e:
    DATABASE_AVAILABLE = False
    print(f"⚠️ 数据库模块导入失败: {e}")
    import traceback
    traceback.print_exc()

# ========================================================
# 🚀 替代 MediaPipe 的高性能人脸检测器 (OpenCV DNN)
# ========================================================
# 只有在 OpenCV 可用时才初始化这些变量
face_net = None

if CV2_AVAILABLE:
    try:
        # 加载 OpenCV 自带的深度学习人脸检测模型（如果可用）
        face_net = None
    except:
        face_net = None


# 为了保证你直接能跑，我们改用一种不需要额外下载 .prototxt 的写法：
def detect_face_dnn(img_rgb):
    """
    使用 OpenCV DNN 进行人脸检测
    """
    if not CV2_AVAILABLE:
        return None
    h, w = img_rgb.shape[:2]
    # 构建 blob - 仅在 OpenCV 可用时执行
    try:
        blob = cv2.dnn.blobFromImage(cv2.resize(img_rgb, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
    except:
        pass
    # 暂时我们用一个最稳妥的逻辑：
    return None  # 占位

# ================ 导入模型 ================
try:
    from models.emotion_model import load_model, EmotionRecognitionModel

    MODEL_IMPORT_SUCCESS = True
    print("✅ 成功导入表情识别模型模块")
except ImportError as e:
    MODEL_IMPORT_SUCCESS = False
    print(f"⚠️  导入模型模块失败: {e}")
    print("将使用模拟模式运行")

# ================ 导入健康建议模块 ================
try:
    from models.health_advisor import HealthAdvisor, EmotionResult

    HEALTH_ADVISOR_IMPORT_SUCCESS = True
    print("✅ 成功导入健康建议模块")
except ImportError as e:
    HEALTH_ADVISOR_IMPORT_SUCCESS = False
    print(f"⚠️  导入健康建议模块失败: {e}")

# ================ 检查阿里云大模型SDK ================
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
    print("✅ OpenAI SDK 可用，支持阿里云百炼平台")
except ImportError as e:
    OPENAI_AVAILABLE = False
    print(f"⚠️  OpenAI SDK 不可用: {e}")
    print("请安装: pip install openai")

# ================ 环境变量检查 ================
print("\n" + "=" * 70)
print("🔑 环境变量检查")
print("=" * 70)

# 检查阿里云百炼API Key
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if dashscope_api_key:
    print(f"✅ 阿里云百炼API Key已设置: {dashscope_api_key[:10]}...")
else:
    print("⚠️  未设置DASHSCOPE_API_KEY环境变量")
    print("   需在环境变量中配置 DASHSCOPE_API_KEY 才能使用大模型功能")

# ================ 初始化Flask应用 ================
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ================ 全局变量 ================
model = None
device = None
transform = None
health_advisor = None

# EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised']
EMOTION_LABELS = ['surprised', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
EMOTION_ZH = {
    'surprised': '惊讶',
    'fear': '恐惧',
    'disgust': '厌恶',
    'happy': '快乐',
    'sad':'悲伤',
    'anger': '愤怒',
    'neutral':'平静'
}

# 创建综合结果目录
COMPREHENSIVE_RESULT_DIR = os.path.join(PROJECT_ROOT, "data", "comprehensive_results")
os.makedirs(COMPREHENSIVE_RESULT_DIR, exist_ok=True)


# ================ 初始化函数 ================
def initialize_model():
    """初始化表情识别模型"""
    global model, device, transform
    print("\n" + "=" * 70)
    print("🤖 初始化表情识别模型")
    print("=" * 70)

    # 设置设备（若 PyTorch 无法导入则降级为 None）
    global device
    if TORCH_AVAILABLE and torch is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {device}")
    else:
        device = None
        print("⚠️ PyTorch 不可用，模型推理功能将被禁用（降级模式）")

    try:
        # 查找模型文件
        possible_paths = [
            os.path.join(PROJECT_ROOT, 'best_model.pth'),
            'best_model.pth',
            os.path.join('models', 'best_model.pth')
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ 找到模型文件: {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")
                break

        if model_path is None:
            print("⚠️  未找到模型文件，创建虚拟模型...")
            if MODEL_IMPORT_SUCCESS:
                if TORCH_AVAILABLE and torch is not None:
                    model = EmotionRecognitionModel(num_classes=7, model_name='resnet18', pretrained=False)
                    model.to(device)
                    model.eval()
                    print("✅ 虚拟模型创建成功")
                else:
                    print("❌ 无法创建虚拟模型：PyTorch 不可用")
                    model = None
            else:
                print("❌ 无法创建模型（模型模块导入失败）")
                model = None
        else:
            # 加载真实模型
            if MODEL_IMPORT_SUCCESS:
                model = load_model(model_path, model_name='resnet18', num_classes=7, device=device)
                print("✅ 模型加载成功")
            else:
                print("❌ 无法加载模型（模型模块导入失败）")
                model = None

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        model = None

    # 定义图像预处理（仅当 torchvision.transforms 可用时创建）
    if transforms is not None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # PyTorch/torchvision 不可用时，提供一个降级的占位 transform（返回原始 PIL Image）
        def _noop_transform(img):
            return img

        transform = _noop_transform
    print("✅ 图像预处理器初始化完成")
    return model is not None


def initialize_health_advisor():
    """初始化健康建议生成器"""
    global health_advisor
    print("\n" + "=" * 70)
    print("🧠 初始化健康建议生成器")
    print("=" * 70)

    try:
        if HEALTH_ADVISOR_IMPORT_SUCCESS:
            # 查找规则文件
            rules_path = os.path.join(PROJECT_ROOT, 'advice_rules.json')
            if os.path.exists(rules_path):
                print(f"✅ 找到规则文件: {rules_path}")
                health_advisor = HealthAdvisor(rules_path=rules_path)
            else:
                print(f"⚠️  未找到规则文件，使用默认规则")
                health_advisor = HealthAdvisor()
            print("✅ 健康建议生成器初始化成功")
            return True
        else:
            print("❌ 健康建议模块导入失败，无法初始化")
            health_advisor = None
            return False
    except Exception as e:
        print(f"❌ 健康建议生成器初始化失败: {e}")
        health_advisor = None
        return False
def predict_emotion(image):
    # 如果模型或 PyTorch 不可用，使用模拟结果返回，保证 API 在降级模式下也能工作
    if model is None or transforms is None or not TORCH_AVAILABLE:
        print("⚠️  使用降级预测：模型或 PyTorch 不可用，返回模拟结果")
        return 'neutral', 0.0, {emo: (1.0 / len(EMOTION_LABELS)) for emo in EMOTION_LABELS}

    try:
        # 1. 转换图像格式
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv2.shape[:2]

        # 2. 如果之前 MediaPipe 报错，我们这里直接用 OpenCV 的底层检测
        # 为了兼容性，我们先尝试最稳的检测
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # 取面积最大的脸
            (x, y, fw, fh) = max(faces, key=lambda b: b[2] * b[3])
            # 扩大人脸框（让模型看全一点）
            pad = int(fw * 0.2)
            face_img = image.crop((max(0, x - pad), max(0, y - pad), min(w, x + fw + pad), min(h, y + fh + pad)))
            print(f"✅ 成功定位人脸区域: {x},{y}")
        else:
            face_img = image
            print("⚠️ 未检测到人脸，使用全图")

        # 3. 模型预测
        img_tensor = transform(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            idx = torch.argmax(probs).item()

        return EMOTION_LABELS[idx], float(probs[idx]), {EMOTION_LABELS[i]: float(probs[i]) for i in
                                                        range(len(EMOTION_LABELS))}
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return 'neutral', 0.0, {}

# ================ 基础API端点 ================
@app.route('/')
def home():
    """API 首页 - 显示所有可用端点"""
    endpoints = {
        '基础端点': {
            'GET /': 'API信息（当前页面）',
            'GET /health': '健康检查',
            'GET /emotions': '获取支持的表情列表'
        },
        '表情识别': {
            'POST /predict': '单张图像表情识别',
            'POST /predict_with_advice': '表情识别+健康建议'
        },
        '摄像头监测': {
            'GET /monitor/status': '获取监测器状态',
            'POST /monitor/start': '开始监测',
            'POST /monitor/pause': '暂停监测',
            'POST /monitor/resume': '继续监测',
            'POST /monitor/stop': '停止监测',
            'GET /monitor/analyze': '分析历史数据'
        },
        'AI大模型分析': {
            'POST /comprehensive_analysis': '综合情绪分析并调用阿里云大模型',
            'POST /multi_agent_analysis': '调用 multiAgent 生成结构化 Markdown 报告'
        },
        'LangGraph Agent (推荐)': {
            'POST /langgraph/chat': '与 LangGraph Agent 对话（首次分析用户性格和近期情绪）',
            'GET /langgraph/status': '获取 LangGraph Agent 状态'
        },
        '数据库操作': {
            'POST /database/save_emotion': '保存情绪记录到数据库',
            'GET /database/status': '获取数据库状态'
        },
        '用户认证': {
            'POST /auth/register': '用户注册',
            'POST /auth/login': '用户登录'
        }
    }

    return jsonify({
        'success': True,
        'message': '表情识别与健康建议 API - LangGraph 增强版',
        'version': '5.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': endpoints,
        'model_loaded': model is not None,
        'health_advisor_loaded': health_advisor is not None,
        'llm_available': OPENAI_AVAILABLE,
        'database_available': DATABASE_AVAILABLE and db_manager is not None,
        'langgraph_available': LANGGRAPH_AVAILABLE and langgraph_agent is not None,
        'device': str(device) if device else 'unknown'
    })


@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'health_advisor_loaded': health_advisor is not None,
        'llm_available': OPENAI_AVAILABLE,
        'device': str(device) if device else 'unknown',
        'timestamp': datetime.now().isoformat(),
        'version': '4.0.0'
    })


@app.route('/emotions', methods=['GET'])
def get_emotions():
    """获取支持的表情列表"""
    emotions_with_zh = [
        {'en': emo, 'zh': EMOTION_ZH.get(emo, emo)}
        for emo in EMOTION_LABELS
    ]

    return jsonify({
        'success': True,
        'emotions': emotions_with_zh,
        'count': len(EMOTION_LABELS),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    单张图像表情识别
    支持格式: multipart/form-data (文件上传) 或 application/json (base64)
    
    请求参数:
    - user_id: 用户ID (可选，默认1)
    - session_id: 会话ID (可选)
    """
    try:
        # 获取用户ID和会话ID
        user_id = 1
        session_id = None
        
        if request.form:
            user_id = int(request.form.get('user_id', 1))
            session_id = request.form.get('session_id')
        elif request.is_json:
            user_id = int(request.json.get('user_id', 1))
            session_id = request.json.get('session_id')

        # 获取图像
        if 'image' in request.files:
            # 从文件上传获取
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        elif request.is_json and 'image' in request.json:
            # 从JSON获取base64编码的图像
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        else:
            return jsonify({
                'success': False,
                'error': '请提供图像文件或base64编码的图像',
                'supported_formats': ['multipart/form-data (文件)', 'application/json (base64)']
            }), 400

        # 预测表情
        emotion, confidence, probabilities = predict_emotion(image)

        # 构建情绪数据
        emotion_data = {
            'emotion': emotion,
            'emotion_zh': EMOTION_ZH.get(emotion, emotion),
            'confidence': float(confidence),
            'probabilities': probabilities
        }

        # 保存到数据库
        record_id = None
        if DATABASE_AVAILABLE and db_manager is not None:
            try:
                record_id = db_manager.save_emotion_record(
                    user_id=user_id,
                    emotion_data=emotion_data,
                    session_id=session_id
                )
                print(f"✅ 情绪记录已保存，记录ID: {record_id}")
            except Exception as db_error:
                print(f"⚠️ 保存到数据库失败: {db_error}")

        return jsonify({
            'success': True,
            'emotion': emotion,
            'emotion_zh': EMOTION_ZH.get(emotion, emotion),
            'confidence': float(confidence),
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat(),
            'record_id': record_id,
            'user_id': user_id
        })

    except Exception as e:
        print(f"❌ 预测错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/predict_with_advice', methods=['POST'])
def predict_with_advice():
    """
    表情识别并生成健康建议
    
    请求参数:
    - user_id: 用户ID (可选，默认1)
    - session_id: 会话ID (可选)
    """
    try:
        # 检查文件
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': '未提供图像文件',
                'timestamp': datetime.now().isoformat()
            }), 400

        file = request.files['image']

        # 检查文件名
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件',
                'timestamp': datetime.now().isoformat()
            }), 400

        # 获取用户ID和会话ID
        user_id = int(request.form.get('user_id', 1))
        session_id = request.form.get('session_id')

        # 获取用户上下文（可选）
        user_context = {}
        if request.form.get('user_context'):
            try:
                user_context = json.loads(request.form['user_context'])
            except:
                user_context = {}

        # 读取图像数据
        image_bytes = file.read()

        # 确保读取到数据
        if not image_bytes:
            return jsonify({
                'success': False,
                'error': '图像文件为空',
                'timestamp': datetime.now().isoformat()
            }), 400

        # 将字节转换为PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 进行情绪预测
        emotion, confidence, probabilities = predict_emotion(image)

        # 构建预测结果
        prediction_result = {
            'emotion': emotion,
            'emotion_zh': EMOTION_ZH.get(emotion, emotion),
            'confidence': float(confidence),
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        }

        # 保存到数据库
        record_id = None
        if DATABASE_AVAILABLE and db_manager is not None:
            try:
                record_id = db_manager.save_emotion_record(
                    user_id=user_id,
                    emotion_data=prediction_result,
                    session_id=session_id
                )
                print(f"✅ 情绪记录已保存，记录ID: {record_id}")
            except Exception as db_error:
                print(f"⚠️ 保存到数据库失败: {db_error}")

        # 生成健康建议
        if health_advisor is not None and HEALTH_ADVISOR_IMPORT_SUCCESS:
            try:
                # 创建情绪结果对象
                emotion_result = EmotionResult(
                    emotion=emotion,
                    confidence=confidence,
                    probabilities=probabilities
                )

                # 生成建议
                health_report = health_advisor.generate_advice(emotion_result, user_context)

                # 合并结果
                full_result = {
                    "success": True,
                    "prediction": prediction_result,
                    "health_advice_report": health_report,
                    "message": "成功生成健康建议",
                    "timestamp": datetime.now().isoformat(),
                    "record_id": record_id,
                    "user_id": user_id
                }

                return jsonify(full_result)

            except Exception as e:
                print(f"❌ 生成健康建议失败: {e}")
                # 返回仅预测结果
                return jsonify({
                    'success': True,
                    'prediction': prediction_result,
                    'health_advice_report': None,
                    'message': '预测成功，但健康建议生成失败',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'record_id': record_id,
                    'user_id': user_id
                })
        else:
            # 健康建议模块未加载
            return jsonify({
                'success': True,
                'prediction': prediction_result,
                'health_advice_report': None,
                'message': '预测成功，但健康建议模块未加载',
                'timestamp': datetime.now().isoformat(),
                'record_id': record_id,
                'user_id': user_id
            })

    except Exception as e:
        print(f"❌ 预测建议错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ================ 新增的API端点：综合情绪分析 ================
@app.route('/comprehensive_analysis', methods=['POST'])
def comprehensive_analysis():
    """
    综合情绪分析并调用阿里云大模型。

    请求体格式:
    {
        "use_history": true,
        "analysis_type": "health_advice",
        "days": 7,
        "user_context": {
            "age_group": "adult",
            "stress_level": "medium",
            "has_support_system": true,
            "is_first_time": false
        }
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是JSON格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json() or {}
        use_history = data.get('use_history', True)
        analysis_type = data.get('analysis_type', 'health_advice')
        days = data.get('days', 7)
        user_context = data.get('user_context') or {
            "age_group": "adult",
            "stress_level": "medium",
            "has_support_system": True,
            "is_first_time": False
        }

        # 读取历史数据
        history_data = []
        if use_history:
            # 尝试多个可能的目录
            possible_dirs = [
                os.path.join(PROJECT_ROOT, "data", "monitor_results", "results"),
                os.path.join(PROJECT_ROOT, "backend", "data", "monitor_results", "results"),
                os.path.join(os.path.dirname(PROJECT_ROOT), "data", "monitor_results", "results"),
            ]
            
            results_dir = None
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    results_dir = dir_path
                    print(f"📂 使用数据目录: {results_dir}")
                    break
            
            if results_dir:
                for filename in os.listdir(results_dir):
                    if not filename.endswith('.json'):
                        continue
                    filepath = os.path.join(results_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                        if days:
                            result_time = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                            if result_time.tzinfo is not None:
                                result_time = result_time.replace(tzinfo=None)
                            cutoff_time = datetime.now() - timedelta(days=days)
                            if result_time < cutoff_time:
                                continue
                        history_data.append(result)
                    except Exception as e:
                        print(f"⚠️  读取监测结果失败 {filename}: {e}")

        # 如果没有历史数据，生成示例数据
        if not history_data:
            history_data = generate_sample_data(10)

        # 分析数据
        analysis_result = analyze_comprehensive_data(history_data, analysis_type)

        # 调用大模型生成建议（如果可用）
        llm_advice = None
        if OPENAI_AVAILABLE and health_advisor:
            try:
                latest_emotion = history_data[-1] if history_data else None
                if latest_emotion:
                    llm_advice = health_advisor.generate_advice(latest_emotion, user_context)
            except Exception as e:
                print(f"⚠️  大模型建议生成失败: {e}")

        return jsonify({
            'success': True,
            'message': '综合分析完成',
            'analysis': analysis_result,
            'llm_advice': llm_advice,
            'total_samples': len(history_data),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 综合分析失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/multi_agent_analysis', methods=['POST'])
def multi_agent_analysis():
    """
    使用 multiAgent 流程进行综合情绪分析，并输出 Markdown 报告。

    请求体格式:
    {
        "days": 7,
        "user_context": {
            "age_group": "adult",
            "stress_level": "medium",
            "has_support_system": true,
            "is_first_time": false
        }
    }
    """
    try:
        if not MULTI_AGENT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'multiAgent 模块不可用，请检查 multiAgent 目录和依赖安装',
                'timestamp': datetime.now().isoformat()
            }), 500

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是JSON格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json() or {}
        days = data.get('days', 7)
        user_context = data.get('user_context') or {
            "age_group": "adult",
            "stress_level": "medium",
            "has_support_system": True,
            "is_first_time": False
        }

        # 尝试多个可能的目录
        possible_dirs = [
            os.path.join(PROJECT_ROOT, "data", "monitor_results", "results"),
            os.path.join(PROJECT_ROOT, "backend", "data", "monitor_results", "results"),
            os.path.join(os.path.dirname(PROJECT_ROOT), "data", "monitor_results", "results"),
        ]
        
        results_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                results_dir = dir_path
                print(f"📂 使用数据目录: {results_dir}")
                break
        
        history_data = []
        if results_dir:
            for filename in os.listdir(results_dir):
                if not filename.endswith('.json'):
                    continue
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    if days:
                        result_time = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                        if result_time.tzinfo is not None:
                            result_time = result_time.replace(tzinfo=None)
                        cutoff_time = datetime.now() - timedelta(days=days)
                        if result_time < cutoff_time:
                            continue
                    history_data.append(result)
                except Exception as e:
                    print(f"❌ 读取监测结果失败 {filename}: {e}")

        if not history_data:
            history_data = generate_sample_data(10)

        output_filename = f"multi_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        output_path = os.path.join(COMPREHENSIVE_RESULT_DIR, output_filename)

        final_report = run_flow_from_data(
            history_data=history_data,
            user_context=user_context,
            output_path=output_path,
        )

        return jsonify({
            'success': True,
            'message': 'multiAgent 分析完成',
            'report_markdown': final_report,
            'output_file': output_filename,
            'total_samples': len(history_data),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ multiAgent 分析失败: {e}")
        error_msg = str(e)
        status_code = 500
        if 'API Key' in error_msg or 'DEEPSEEK_API_KEY' in error_msg or 'DASHSCOPE_API_KEY' in error_msg:
            status_code = 400
            error_msg = f"{error_msg} 请先配置环境变量后重启后端服务。"
        return jsonify({
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }), status_code


def generate_sample_data(num_samples):
    """生成示例数据"""
    sample_data = []
    emotions = ['happy', 'sad', 'anger', 'fear', 'surprised', 'disgust', 'neutral']

    for i in range(num_samples):
        emotion = np.random.choice(emotions)
        confidence = np.random.uniform(0.6, 0.95)

        # 生成概率分布
        probabilities = {}
        for e in emotions:
            if e == emotion:
                probabilities[e] = confidence
            else:
                probabilities[e] = (1 - confidence) / (len(emotions) - 1)

        sample_data.append({
            'timestamp': (datetime.now() - timedelta(hours=i * 2)).isoformat(),
            'emotion': emotion,
            'emotion_zh': EMOTION_ZH.get(emotion, emotion),
            'confidence': confidence,
            'probabilities': probabilities
        })

    return sample_data


def analyze_comprehensive_data(history_data, analysis_type):
    """分析综合数据"""
    # 情绪分布统计
    emotion_counts = {}
    emotion_confidences = []
    timestamps = []

    for result in history_data:
        emotion = result.get('emotion', 'unknown')
        confidence = result.get('confidence', 0)

        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
        emotion_counts[emotion] += 1
        emotion_confidences.append(confidence)
        timestamps.append(result.get('timestamp', ''))

    # 计算平均置信度
    avg_confidence = sum(emotion_confidences) / len(emotion_confidences) if emotion_confidences else 0

    # 情绪趋势分析
    emotion_trend = {}
    if len(timestamps) > 1:
        try:
            # 按时间排序
            sorted_data = sorted(history_data, key=lambda x: x.get('timestamp', ''))
            # 提取时间序列的情绪变化
            emotion_timeseries = []
            for item in sorted_data:
                emotion_timeseries.append({
                    'time': item.get('timestamp', ''),
                    'emotion': item.get('emotion', 'unknown'),
                    'confidence': item.get('confidence', 0)
                })
            emotion_trend['timeseries'] = emotion_timeseries
        except Exception as e:
            print(f"⚠️  情绪趋势分析失败: {e}")

    # 主要情绪
    if emotion_counts:
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
    else:
        dominant_emotion = ('unknown', 0)

    # 情绪稳定性计算（基于情绪变化的频率）
    stability_score = 0
    if len(history_data) > 1:
        emotion_changes = 0
        for i in range(1, len(history_data)):
            if history_data[i].get('emotion') != history_data[i - 1].get('emotion'):
                emotion_changes += 1
        stability_score = max(0, 100 - (emotion_changes / len(history_data) * 100))

    # 生成分析结果
    analysis = {
        'emotion_distribution': emotion_counts,
        'total_samples': len(history_data),
        'average_confidence': avg_confidence,
        'dominant_emotion': {
            'emotion': dominant_emotion[0],
            'emotion_zh': EMOTION_ZH.get(dominant_emotion[0], dominant_emotion[0]),
            'count': dominant_emotion[1],
            'percentage': (dominant_emotion[1] / len(history_data)) * 100 if history_data else 0
        },
        'time_range': {
            'start': min(timestamps) if timestamps else '',
            'end': max(timestamps) if timestamps else ''
        },
        'emotion_trend': emotion_trend,
        'stability_score': stability_score,
        'analysis_type': analysis_type
    }

    # 根据分析类型添加特定信息
    if analysis_type == 'health_advice':
        analysis['health_insights'] = generate_health_insights(emotion_counts, len(history_data))
    elif analysis_type == 'monitor_analysis':
        analysis['monitor_stats'] = generate_monitor_stats(history_data)
    elif analysis_type == 'detailed_report':
        analysis['detailed_analysis'] = generate_detailed_analysis(history_data)

    return analysis


def generate_health_insights(emotion_counts, total_samples):
    """生成健康洞察"""
    insights = []

    # 负面情绪比例
    negative_emotions = ['anger', 'sad', 'fear']
    negative_count = sum(emotion_counts.get(emo, 0) for emo in negative_emotions)
    negative_percentage = (negative_count / total_samples) * 100 if total_samples > 0 else 0

    if negative_percentage > 50:
        insights.append(f"负面情绪比例较高 ({negative_percentage:.1f}%)，建议关注情绪管理")
    elif negative_percentage > 30:
        insights.append(f"负面情绪比例适中 ({negative_percentage:.1f}%)，建议保持关注")
    else:
        insights.append(f"负面情绪比例较低 ({negative_percentage:.1f}%)，情绪状态良好")

    # 正面情绪
    positive_emotions = ['happy']
    positive_count = sum(emotion_counts.get(emo, 0) for emo in positive_emotions)
    positive_percentage = (positive_count / total_samples) * 100 if total_samples > 0 else 0

    insights.append(f"快乐情绪占比 {positive_percentage:.1f}%")

    # 情绪多样性
    emotion_diversity = len(emotion_counts)
    if emotion_diversity > 4:
        insights.append("情绪体验丰富，情感表达多样")
    elif emotion_diversity > 2:
        insights.append("情绪体验较为丰富")
    else:
        insights.append("情绪体验较为单一")

    return insights


def generate_monitor_stats(history_data):
    """生成监测统计"""
    stats = {
        'total_captures': len(history_data),
        'time_period': '未知',
        'capture_frequency': '未知',
        'success_rate': '100%'
    }

    if len(history_data) > 1:
        try:
            # 计算时间范围
            timestamps = [datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) for r in history_data]
            time_diff = max(timestamps) - min(timestamps)
            stats['time_period'] = f"{time_diff.days}天{time_diff.seconds // 3600}小时"

            # 计算平均捕获频率
            if len(timestamps) > 2:
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
                    intervals.append(interval)
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval < 60:
                    stats['capture_frequency'] = f"平均每{avg_interval:.0f}秒一次"
                else:
                    stats['capture_frequency'] = f"平均每{avg_interval / 60:.1f}分钟一次"
        except:
            pass

    return stats


def generate_detailed_analysis(history_data):
    """生成详细分析"""
    analysis = {
        'emotion_transitions': [],
        'confidence_analysis': {
            'min': 0,
            'max': 0,
            'avg': 0
        },
        'patterns': [],
        'recommendations': []
    }

    if history_data:
        # 计算置信度统计
        confidences = [r.get('confidence', 0) for r in history_data]
        analysis['confidence_analysis'] = {
            'min': min(confidences),
            'max': max(confidences),
            'avg': sum(confidences) / len(confidences)
        }

        # 检测情绪转换
        if len(history_data) > 1:
            emotions = [r.get('emotion', 'unknown') for r in history_data]
            transitions = []
            for i in range(1, len(emotions)):
                if emotions[i] != emotions[i - 1]:
                    from_emo = EMOTION_ZH.get(emotions[i - 1], emotions[i - 1])
                    to_emo = EMOTION_ZH.get(emotions[i], emotions[i])
                    transitions.append(f"{from_emo} → {to_emo}")
            analysis['emotion_transitions'] = transitions[:5]  # 只显示前5个

        # 检测模式
        emotion_counts = {}
        for result in history_data:
            emotion = result.get('emotion', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # 识别常见模式
        total = len(history_data)
        for emotion, count in emotion_counts.items():
            percentage = (count / total) * 100
            if percentage > 60:
                analysis['patterns'].append(f"主要情绪模式: {EMOTION_ZH.get(emotion, emotion)} (占比{percentage:.1f}%)")
            elif percentage > 30:
                analysis['patterns'].append(f"次要情绪模式: {EMOTION_ZH.get(emotion, emotion)} (占比{percentage:.1f}%)")

        # 生成建议
        if analysis['confidence_analysis']['avg'] > 0.8:
            analysis['recommendations'].append("情绪识别置信度高，分析结果可靠")
        else:
            analysis['recommendations'].append("情绪识别置信度中等，建议多次监测获取更准确结果")

        if len(analysis['emotion_transitions']) > 5:
            analysis['recommendations'].append("情绪变化频繁，建议关注情绪稳定性")

    return analysis


def generate_algorithm_based_analysis(analysis_result, analysis_type, user_context=None):
    """使用算法生成分析报告（当大模型不可用时）"""

    emotion_counts = analysis_result.get('emotion_distribution', {})
    total_samples = analysis_result.get('total_samples', 0)
    dominant_emotion = analysis_result.get('dominant_emotion', {})
    stability_score = analysis_result.get('stability_score', 0)

    dominant_emotion_name = dominant_emotion.get('emotion_zh', '未知')
    dominant_percentage = dominant_emotion.get('percentage', 0)

    # 计算负面情绪比例
    negative_emotions = ['anger', 'sad', 'fear']
    negative_count = sum(emotion_counts.get(emo, 0) for emo in negative_emotions)
    negative_percentage = (negative_count / total_samples) * 100 if total_samples > 0 else 0

    # 计算正面情绪比例
    positive_emotions = ['happy']
    positive_count = sum(emotion_counts.get(emo, 0) for emo in positive_emotions)
    positive_percentage = (positive_count / total_samples) * 100 if total_samples > 0 else 0

    # 根据用户上下文调整分析
    age_group = user_context.get('age_group', 'adult') if user_context else 'adult'
    stress_level = user_context.get('stress_level', 'medium') if user_context else 'medium'
    has_support = user_context.get('has_support_system', True) if user_context else True
    is_first_time = user_context.get('is_first_time', False) if user_context else False

    # 根据年龄组调整建议
    age_specific_advice = []
    if age_group == 'child':
        age_specific_advice.extend([
            "建议家长多关注孩子的情绪变化",
            "鼓励孩子用绘画或游戏表达情绪",
            "建立规律的作息和情绪表达习惯"
        ])
    elif age_group == 'teen':
        age_specific_advice.extend([
            "青春期的情绪波动是正常的",
            "鼓励与朋友或家长分享感受",
            "培养健康的兴趣爱好"
        ])
    elif age_group == 'elder':
        age_specific_advice.extend([
            "保持社交活动，避免孤独",
            "适度运动，保持身心健康",
            "参与社区活动或志愿工作"
        ])

    # 根据压力水平调整建议
    stress_specific_advice = []
    if stress_level == 'high':
        stress_specific_advice.extend([
            "高压力状态下需要特别注意情绪调节",
            "每天安排15-30分钟的放松时间",
            "学习压力管理技巧"
        ])

    # 根据支持系统调整建议
    support_specific_advice = []
    if not has_support:
        support_specific_advice.extend([
            "建议主动建立社交支持网络",
            "参加兴趣小组或社区活动",
            "考虑寻求专业心理咨询"
        ])
    else:
        support_specific_advice.extend([
            "善用现有的社会支持系统",
            "与信任的人分享情绪体验",
            "在需要时寻求亲友帮助"
        ])

    # 根据使用经验调整建议
    experience_specific_advice = []
    if is_first_time:
        experience_specific_advice.extend([
            "欢迎首次使用情绪监测系统",
            "建议连续使用1-2周建立情绪基线",
            "记录情绪触发事件以便分析"
        ])
    else:
        experience_specific_advice.extend([
            "继续坚持情绪监测和记录",
            "对比历史数据观察变化趋势",
            "根据建议调整情绪管理策略"
        ])

    if analysis_type == 'health_advice':
        # 健康建议算法
        risk_level = "低"
        if negative_percentage > 50:
            risk_level = "高"
        elif negative_percentage > 30:
            risk_level = "中"

        advice = {
            'overall_assessment': f"基于{total_samples}条数据分析，主要情绪为{dominant_emotion_name} (占比{dominant_percentage:.1f}%)",
            'risk_assessment': {
                'level': risk_level,
                'description': f"负面情绪比例{negative_percentage:.1f}%，情绪稳定性{stability_score:.1f}%",
                'negative_percentage': negative_percentage,
                'positive_percentage': positive_percentage
            },
            'immediate_actions': [
                "深呼吸10次，放松身心",
                "短暂离开当前环境，转移注意力",
                "喝一杯温水，平复情绪",
                "进行2分钟的拉伸运动"
            ],
            'daily_tips': [
                              "每天记录情绪变化，识别触发因素",
                              "保持规律作息，保证7-8小时睡眠",
                              "每天进行30分钟有氧运动",
                              "练习正念冥想或呼吸训练"
                          ] + age_specific_advice[:2] + stress_specific_advice[:2],
            'long_term_suggestions': [
                                         "学习情绪管理技巧（如认知行为疗法）",
                                         "建立健康的生活习惯",
                                         "培养积极的思维方式",
                                         "定期进行心理健康自我评估"
                                     ] + support_specific_advice + experience_specific_advice,
            'user_context_notes': {
                'age_group': age_group,
                'stress_level': stress_level,
                'has_support_system': has_support,
                'is_first_time': is_first_time
            },
            'algorithm_based': True
        }
        return advice

    elif analysis_type == 'monitor_analysis':
        # 监测分析算法
        stability_text = "稳定" if stability_score > 70 else ("中等" if stability_score > 40 else "波动大")
        frequency_advice = "建议保持当前监测频率" if stability_score > 60 else "建议增加监测频率以获得更准确数据"

        analysis = {
            'summary': f"监测数据分析报告 - 共{total_samples}条数据",
            'key_findings': [
                f"主要情绪: {dominant_emotion_name} (占比{dominant_percentage:.1f}%)",
                f"情绪稳定性: {stability_score:.1f}% ({stability_text})",
                f"负面情绪比例: {negative_percentage:.1f}%",
                f"正面情绪比例: {positive_percentage:.1f}%"
            ],
            'monitoring_insights': [
                f"数据采集数量: {total_samples}条",
                f"情绪多样性: {len(emotion_counts)}种不同情绪",
                f"主要情绪持续性: {'较强' if dominant_percentage > 50 else '一般'}"
            ],
            'user_specific_recommendations': age_specific_advice + support_specific_advice + experience_specific_advice,
            'monitoring_recommendations': [
                frequency_advice,
                "在不同时间段进行监测以获得更全面的数据",
                "记录情绪触发事件以便分析",
                "设置情绪变化提醒和预警"
            ],
            'algorithm_based': True
        }
        return analysis

    elif analysis_type == 'detailed_report':
        # 详细报告算法
        emotion_diversity = len(emotion_counts)
        diversity_text = "丰富" if emotion_diversity > 4 else ("适中" if emotion_diversity > 2 else "单一")

        # 情绪转换分析
        emotion_transitions = []
        if total_samples > 1:
            try:
                # 获取历史数据中的情绪序列
                emotions_sequence = []
                if analysis_result.get('emotion_trend', {}).get('timeseries'):
                    for item in analysis_result['emotion_trend']['timeseries']:
                        emotions_sequence.append(item.get('emotion', 'unknown'))

                if len(emotions_sequence) > 1:
                    for i in range(1, len(emotions_sequence)):
                        if emotions_sequence[i] != emotions_sequence[i - 1]:
                            from_emo = EMOTION_ZH.get(emotions_sequence[i - 1], emotions_sequence[i - 1])
                            to_emo = EMOTION_ZH.get(emotions_sequence[i], emotions_sequence[i])
                            emotion_transitions.append(f"{from_emo} → {to_emo}")
            except:
                pass

        report = {
            'title': f"详细情绪分析报告 - {total_samples}条数据",
            'executive_summary': f"基于数据分析，您的主要情绪模式为{dominant_emotion_name}，情绪体验{diversity_text}，稳定性{stability_score:.1f}%。",
            'demographic_considerations': f"考虑到您的年龄组为{age_group}，压力水平为{stress_level}，以下建议针对您的个人情况定制。",
            'detailed_analysis': [
                f"情绪分布: {', '.join([f'{EMOTION_ZH.get(e, e)}: {c}次 ({c / total_samples * 100:.1f}%)' for e, c in emotion_counts.items()])}",
                f"情绪稳定性评分: {stability_score:.1f}/100",
                f"情绪多样性: {emotion_diversity}种不同情绪",
                f"负面情绪比例: {negative_percentage:.1f}%",
                f"正面情绪比例: {positive_percentage:.1f}%"
            ],
            'pattern_analysis': [
                f"主要情绪模式: {dominant_emotion_name} (出现{dominant_emotion.get('count', 0)}次)",
                f"情绪变化频率: {len(emotion_transitions)}次明显情绪转换" if emotion_transitions else "情绪变化较少",
                f"情绪转换模式: {'、'.join(emotion_transitions[:3])}" if emotion_transitions else "无明显转换模式"
            ],
            'professional_insights': [
                                         "情绪模式显示相对稳定，建议继续保持情绪管理习惯",
                                         "如有负面情绪持续出现，建议关注并适当调整应对策略",
                                         "情绪多样性有助于情感健康发展和心理弹性提升"
                                     ] + age_specific_advice,
            'personalized_recommendations': [
                                                "建立个人情绪日记，记录每日情绪变化及原因",
                                                "学习正念冥想技巧，提高情绪觉察和自我调节能力",
                                                "根据情绪模式调整日常生活节奏和活动安排",
                                                "定期回顾情绪数据，识别改善趋势和潜在问题"
                                            ] + support_specific_advice + stress_specific_advice,
            'follow_up_suggestions': [
                                         "建议每2周进行一次综合情绪分析",
                                         "在情绪波动较大时增加监测频率",
                                         "将情绪数据与生活事件关联分析",
                                         "考虑与心理健康专业人士分享分析结果"
                                     ] + experience_specific_advice,
            'algorithm_based': True
        }
        return report

    return {"error": "未知分析类型", "algorithm_based": True}


# ================ 修改 call_aliyun_llm 函数定义 ================
def call_aliyun_llm(analysis_result, analysis_type, user_context=None):
    """
    调用单智能体大模型：优先 DeepSeek，其次阿里云百炼。
    使用兼容 OpenAI 的 SDK。
    """
    try:
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK 未安装，无法调用大模型")

        # 读取 API Key（支持环境变量和 API_Key.json）
        deepseek_key = None
        dashscope_key = None
        if get_api_key:
            deepseek_key = get_api_key("deepseek")
            dashscope_key = get_api_key("dashscope")

        if not deepseek_key:
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not dashscope_key:
            dashscope_key = os.getenv("DASHSCOPE_API_KEY")

        if not deepseek_key and not dashscope_key:
            raise ValueError("未找到可用 API Key，请配置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY（环境变量或 API_Key.json）")

        # 优先 DeepSeek，回退百炼
        if deepseek_key:
            api_key = deepseek_key
            base_url = "https://api.deepseek.com"
            model_name = "deepseek-chat"
            provider_name = "DeepSeek"
        else:
            api_key = dashscope_key
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model_name = "qwen-plus"
            provider_name = "DashScope"

        # 初始化OpenAI客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        print(f"📤 调用单智能体大模型 [{provider_name}]，分析类型: {analysis_type}")

        # 构建提示词 - 添加 user_context 参数
        prompt = build_llm_prompt(analysis_result, analysis_type, user_context)

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(analysis_type, user_context)
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # 调用模型
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )

        # 获取回复
        llm_response = completion.choices[0].message.content

        print(f"✅ 大模型调用成功 [{provider_name}]，回复长度: {len(llm_response)}")

        return {
            'success': True,
            'provider': provider_name,
            'model': model_name,
            'raw_response': llm_response,
            'usage': {
                'total_tokens': completion.usage.total_tokens if completion.usage else 0,
                'prompt_tokens': completion.usage.prompt_tokens if completion.usage else 0,
                'completion_tokens': completion.usage.completion_tokens if completion.usage else 0
            }
        }

    except Exception as e:
        print(f"❌ 调用单智能体大模型失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback': generate_algorithm_based_analysis(analysis_result, analysis_type)
        }


def build_system_prompt(analysis_type, user_context=None):
    """构建系统提示词"""

    # 基础系统提示
    base_prompts = {
        'health_advice': """你是一位专业的心理健康顾问，擅长情绪分析和健康建议。
请根据用户提供的情绪识别数据和用户个人信息，提供专业、具体、可操作的健康建议。
你的建议应该：
1. 基于数据事实，客观分析
2. 结合用户的具体情况提供个性化建议
3. 提供具体可操作的建议
4. 考虑不同情绪状态的心理影响
5. 必要时提供紧急建议
    6. 使用中文回复，结构清晰，易于理解
    7. 所有输出必须使用 Markdown 格式，至少包含以下二级标题：
       - ## 风险评估
       - ## 核心建议
       - ## 今日行动清单""",

        'monitor_analysis': """你是一位专业的情绪数据分析师，擅长情绪监测数据分析。
请根据用户提供的情绪监测数据，提供专业的分析报告。
你的分析应该：
1. 基于数据趋势，客观分析
2. 识别异常模式和潜在问题
3. 考虑用户的具体情况提供监测建议
4. 预测可能的情绪变化趋势
    5. 使用中文回复，结构清晰，数据驱动
    6. 所有输出必须使用 Markdown 格式，至少包含以下二级标题：
       - ## 趋势摘要
       - ## 关键发现
       - ## 后续监测建议""",

        'detailed_report': """你是一位资深的心理学专家，擅长情绪模式识别和深度分析。
请根据用户提供的详细情绪数据，提供专业的心理学分析报告。
你的报告应该：
1. 深入分析情绪模式和背后的心理因素
2. 结合用户情况提供专业解读
3. 提供个性化的发展建议
4. 关注情绪健康和长期发展
5. 使用中文回复，专业且有深度"""
    }

    system_prompt = base_prompts.get(analysis_type,
                                     "你是一位专业的心理健康顾问，请根据用户提供的信息提供专业的分析和建议。")

    # 如果有用户上下文，添加到系统提示词中
    if user_context and isinstance(user_context, dict) and len(user_context) > 0:
        user_info = "\n\n## 用户个人信息\n"

        # 年龄组映射
        if user_context.get('age_group'):
            age_groups = {
                'child': '儿童 (0-12岁)',
                'teen': '青少年 (13-19岁)',
                'adult': '成人 (20-59岁)',
                'elder': '老年人 (60岁以上)'
            }
            user_info += f"- 年龄组: {age_groups.get(user_context['age_group'], user_context['age_group'])}\n"

        # 压力水平映射
        if user_context.get('stress_level'):
            stress_levels = {
                'low': '低压力',
                'medium': '中等压力',
                'high': '高压力'
            }
            user_info += f"- 压力水平: {stress_levels.get(user_context['stress_level'], user_context['stress_level'])}\n"

        # 支持系统
        if user_context.get('has_support_system') is not None:
            has_support = user_context['has_support_system']
            user_info += f"- 社会支持系统: {'有 (有亲友支持)' if has_support else '无 (缺乏社会支持)'}\n"

        # 是否首次使用
        if user_context.get('is_first_time') is not None:
            is_first_time = user_context['is_first_time']
            user_info += f"- 使用经验: {'首次使用情绪监测系统' if is_first_time else '有情绪监测经验'}\n"

        # 职业类型（可选字段）
        if user_context.get('occupation'):
            occupations = {
                'student': '学生',
                'office_worker': '上班族',
                'freelancer': '自由职业者',
                'retired': '退休人员'
            }
            user_info += f"- 职业类型: {occupations.get(user_context['occupation'], user_context['occupation'])}\n"

        # 近期活动（可选字段）
        if user_context.get('recent_activity'):
            user_info += f"- 近期主要活动: {user_context['recent_activity']}\n"

        user_info += "\n请根据以上用户信息，结合情绪数据分析结果，提供更加个性化、针对性的分析和建议。"

        system_prompt += user_info

    return system_prompt




def build_llm_prompt(analysis_result, analysis_type, user_context=None):
    """构建大模型提示词"""

    prompt = f"""
    请根据以下情绪识别数据分析结果，提供专业的分析和建议：

    ## 数据分析摘要
    - 总样本数: {analysis_result.get('total_samples', 0)}
    - 情绪分布: {json.dumps(analysis_result.get('emotion_distribution', {}), ensure_ascii=False)}
    - 主要情绪: {analysis_result.get('dominant_emotion', {}).get('emotion_zh', '未知')} (占比{analysis_result.get('dominant_emotion', {}).get('percentage', 0):.1f}%)
    - 平均置信度: {analysis_result.get('average_confidence', 0):.2%}
    - 情绪稳定性: {analysis_result.get('stability_score', 0):.1f}%
    - 时间范围: {analysis_result.get('time_range', {}).get('start', '未知')} 至 {analysis_result.get('time_range', {}).get('end', '未知')}

    """

    # 添加健康洞察（如果有）
    if analysis_type == 'health_advice' and 'health_insights' in analysis_result:
        prompt += f"## 健康洞察\n"
        for insight in analysis_result['health_insights']:
            prompt += f"- {insight}\n"
        prompt += "\n"

    if analysis_type == 'health_advice':
        prompt += """
        请根据以上数据，提供：
        1. 整体情绪状态评估
        2. 具体的健康建议（立即行动、日常贴士、长期建议）
        3. 风险评估
        4. 如有需要，提供紧急建议
        """
    elif analysis_type == 'monitor_analysis':
        prompt += """
        请根据以上数据，提供：
        1. 监测数据分析
        2. 情绪变化趋势
        3. 异常模式检测
        4. 监测建议
        """
    elif analysis_type == 'detailed_report':
        prompt += """
        请根据以上数据，提供：
        1. 详细情绪分析报告
        2. 情绪模式识别
        3. 专业心理学解读
        4. 个性化建议
        """

    prompt += """

    请用中文回复，结构清晰，专业且易于理解。
    """

    return prompt


# ================ 摄像头监测API端点 ================
# 首先，如果导入摄像头监测模块失败，创建一个虚拟的监测器
try:
    # 尝试从api目录导入
    from api.camera_monitor import get_monitor

    CAMERA_MONITOR_IMPORT_SUCCESS = True
    print("✅ 成功导入摄像头监测模块")
except Exception as e:
    # 捕获所有异常（包括 OSError/DLL 加载错误），避免因依赖不可用而使整个 API 崩溃
    print(f"⚠️  导入摄像头监测模块失败: {e}")
    print("   创建虚拟摄像头监测模块")

    CAMERA_MONITOR_IMPORT_SUCCESS = False


    # 创建虚拟的摄像头监测器类
    class VirtualCameraMonitor:
        def __init__(self, model_path=None, save_dir="monitor_results"):
            self.model_path = model_path
            self.save_dir = save_dir
            self.is_monitoring = False
            self.total_captures = 0
            self.successful_analyses = 0
            self.capture_interval = 5
            self.camera_index = 0

            # 创建保存目录
            results_dir = os.path.join(save_dir, "results")
            os.makedirs(results_dir, exist_ok=True)

        def get_status(self):
            return {
                "is_monitoring": self.is_monitoring,
                "total_captures": self.total_captures,
                "successful_analyses": self.successful_analyses,
                "capture_interval": self.capture_interval,
                "camera_index": self.camera_index
            }

        def start(self, camera_index=0, capture_interval=5):
            self.camera_index = camera_index
            self.capture_interval = capture_interval
            self.is_monitoring = True

            # 模拟摄像头启动
            print(f"📷 虚拟摄像头监测已启动 (摄像头索引: {camera_index}, 抓拍间隔: {capture_interval}秒)")

            return {"status": "started", "message": "虚拟监测模式 - 摄像头功能需要安装OpenCV"}

        def pause(self):
            self.is_monitoring = False
            return {"status": "paused"}

        def resume(self):
            self.is_monitoring = True
            return {"status": "resumed"}

        def stop(self):
            self.is_monitoring = False
            return {"status": "stopped"}

        def analyze_history(self):
            # 模拟分析历史数据
            return {"status": "analysis_completed", "message": "虚拟分析完成"}


    def get_monitor(model_path=None, save_dir="monitor_results"):
        return VirtualCameraMonitor(model_path=model_path, save_dir=save_dir)

monitor = None


def initialize_camera_monitor(user_id=1, session_id=None):
    """初始化摄像头监测器"""
    global monitor
    print("\n" + "=" * 70)
    print("📷 初始化摄像头监测器")
    print("=" * 70)

    try:
        # 设置保存目录
        save_dir = os.path.join(PROJECT_ROOT, "data", "monitor_results")

        # 查找模型文件
        model_path = None
        possible_paths = [
            os.path.join(PROJECT_ROOT, 'best_model.pth'),
            os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ 找到监测模型: {path}")
                break

        monitor = get_monitor(
            model_path=model_path, 
            save_dir=save_dir,
            user_id=user_id,
            session_id=session_id,
            db_manager=db_manager if DATABASE_AVAILABLE else None
        )
        print("✅ 摄像头监测器初始化成功")
        return True
    except Exception as e:
        print(f"❌ 摄像头监测器初始化失败: {e}")
        monitor = None
        return False


@app.route('/monitor/status', methods=['GET'])
def get_monitor_status():
    """获取监测器状态"""
    try:
        if monitor is None:
            return jsonify({
                'success': False,
                'error': '监测器未初始化',
                'timestamp': datetime.now().isoformat()
            }), 500

        status = monitor.get_status()

        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"❌ 获取监测器状态失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/monitor/start', methods=['POST'])
def start_monitor():
    """开始摄像头监测"""
    try:
        # 获取请求参数
        user_id = 1
        session_id = None
        if request.is_json:
            data = request.get_json()
            camera_index = data.get('camera_index', 0)
            capture_interval = data.get('capture_interval', 5)
            user_id = data.get('user_id', 1)
            session_id = data.get('session_id')
        else:
            camera_index = request.form.get('camera_index', 0, type=int)
            capture_interval = request.form.get('capture_interval', 5, type=int)
            user_id = int(request.form.get('user_id', 1))
            session_id = request.form.get('session_id')

        # 如果监测器未初始化，先初始化
        if monitor is None:
            initialize_camera_monitor(user_id=user_id, session_id=session_id)
        else:
            # 更新监测器的用户信息
            monitor.user_id = user_id
            monitor.session_id = session_id
            monitor.db_manager = db_manager if DATABASE_AVAILABLE else None

        # 检查OpenCV是否可用
        if not CV2_AVAILABLE:
            print("⚠️  OpenCV不可用，使用虚拟监测模式")

        # 调用监测器的start方法
        result = monitor.start(camera_index=camera_index, capture_interval=capture_interval)

        return jsonify({
            'success': True,
            'message': '摄像头监测已启动',
            'camera_index': camera_index,
            'capture_interval': capture_interval,
            'user_id': user_id,
            'session_id': session_id,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 启动监测器失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/monitor/pause', methods=['POST'])
def pause_monitor():
    """暂停摄像头监测"""
    try:
        if monitor is None:
            return jsonify({
                'success': False,
                'error': '监测器未初始化',
                'timestamp': datetime.now().isoformat()
            }), 500

        result = monitor.pause()
        return jsonify({
            'success': True,
            'message': '摄像头监测已暂停',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 暂停监测器失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/monitor/resume', methods=['POST'])
def resume_monitor():
    """继续摄像头监测"""
    try:
        if monitor is None:
            return jsonify({
                'success': False,
                'error': '监测器未初始化',
                'timestamp': datetime.now().isoformat()
            }), 500

        result = monitor.resume()
        return jsonify({
            'success': True,
            'message': '摄像头监测已继续',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 继续监测器失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/monitor/stop', methods=['POST'])
def stop_monitor():
    """停止摄像头监测"""
    try:
        if monitor is None:
            return jsonify({
                'success': False,
                'error': '监测器未初始化',
                'timestamp': datetime.now().isoformat()
            }), 500

        result = monitor.stop()
        return jsonify({
            'success': True,
            'message': '摄像头监测已停止',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 停止监测器失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/monitor/analyze', methods=['GET'])
def analyze_monitor_data():
    """分析监测历史数据"""
    try:
        # 如果没有监测器，尝试从保存的文件中分析
        results_dir = os.path.join(PROJECT_ROOT, "data", "monitor_results", "results")
        if not os.path.exists(results_dir):
            return jsonify({
                'success': False,
                'error': '没有找到监测历史数据',
                'timestamp': datetime.now().isoformat()
            }), 404

        # 收集所有结果文件
        results = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        results.append(result)
                except Exception as e:
                    print(f"❌ 读取结果文件失败 {filename}: {e}")
                    continue

        if not results:
            return jsonify({
                'success': False,
                'error': '没有有效的监测数据',
                'timestamp': datetime.now().isoformat()
            }), 404

        # 分析数据
        emotion_counts = {}
        emotion_confidences = {}
        total_results = len(results)

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
            avg_confidences[emotion] = sum(conf_list) / len(conf_list)

        # 找到主要情绪
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            dominant_emotion_name = dominant_emotion[0]
            dominant_count = dominant_emotion[1]
            dominant_percentage = (dominant_count / total_results) * 100
        else:
            dominant_emotion_name = 'unknown'
            dominant_percentage = 0

        analysis = {
            'emotion_distribution': emotion_counts,
            'average_confidences': avg_confidences,
            'dominant_emotion': {
                'emotion': dominant_emotion_name,
                'emotion_zh': EMOTION_ZH.get(dominant_emotion_name, dominant_emotion_name),
                'count': dominant_count if emotion_counts else 0,
                'percentage': dominant_percentage
            },
            'total_results': total_results
        }

        return jsonify({
            'success': True,
            'analysis': analysis,
            'total_results': total_results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 分析监测数据失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ================ LangGraph Agent API 端点 ================
@app.route('/langgraph/chat', methods=['POST'])
def langgraph_chat():
    """
    与 LangGraph Agent 对话
    首次会话会分析用户性格（半年-一年）和近期情绪（1-3个月）
    之后使用缓存的系统提示词
    
    请求体格式:
    {
        "message": "用户消息",
        "user_id": 1,  // 可选，默认为1
        "session_id": "session_123"  // 可选，自动生成
    }
    """
    try:
        if not LANGGRAPH_AVAILABLE or langgraph_agent is None:
            return jsonify({
                'success': False,
                'error': 'LangGraph Agent 不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是 JSON 格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', 1)
        session_id = data.get('session_id')

        if not user_message:
            return jsonify({
                'success': False,
                'error': '请提供消息内容',
                'timestamp': datetime.now().isoformat()
            }), 400

        # 与 LangGraph Agent 对话
        result = langgraph_agent.chat(user_message, user_id, session_id)

        return jsonify(result)

    except Exception as e:
        print(f"❌ LangGraph Agent 对话失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/langgraph/status', methods=['GET'])
def langgraph_status():
    """获取 LangGraph Agent 状态"""
    return jsonify({
        'success': True,
        'available': LANGGRAPH_AVAILABLE and langgraph_agent is not None,
        'database_available': DATABASE_AVAILABLE and db_manager is not None,
        'timestamp': datetime.now().isoformat()
    })


# ================ 数据库 API 端点 ================
@app.route('/database/save_emotion', methods=['POST'])
def save_emotion_to_db():
    """保存情绪记录到数据库"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是 JSON 格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json()
        emotion_data = data.get('emotion_data')
        user_id = data.get('user_id', 1)
        session_id = data.get('session_id')

        if not emotion_data:
            return jsonify({
                'success': False,
                'error': '请提供情绪数据',
                'timestamp': datetime.now().isoformat()
            }), 400

        record_id = db_manager.save_emotion_record(user_id, emotion_data, session_id)

        return jsonify({
            'success': True,
            'record_id': record_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 保存情绪记录失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/database/status', methods=['GET'])
def database_status():
    """获取数据库状态"""
    return jsonify({
        'success': True,
        'available': DATABASE_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


# ================ 用户认证 API 端点 ================
@app.route('/auth/register', methods=['POST'])
def register():
    """
    用户注册
    
    请求体格式:
    {
        "username": "用户名",
        "password": "密码",
        "email": "邮箱（可选）",
        "age_group": "年龄组（可选，child/teen/adult/elder）",
        "stress_level": "压力水平（可选，low/medium/high）",
        "has_support_system": true/false（可选）
    }
    """
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是 JSON 格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        age_group = data.get('age_group', 'adult')
        stress_level = data.get('stress_level', 'medium')
        has_support_system = data.get('has_support_system', True)

        if not username or not password:
            return jsonify({
                'success': False,
                'error': '用户名和密码不能为空',
                'timestamp': datetime.now().isoformat()
            }), 400

        user = db_manager.register_user(
            username=username,
            password=password,
            email=email,
            age_group=age_group,
            stress_level=stress_level,
            has_support_system=has_support_system
        )

        if user:
            # 不返回密码
            user_data = {k: v for k, v in user.items() if k != 'password'}
            return jsonify({
                'success': True,
                'message': '注册成功',
                'user': user_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': '用户名已存在',
                'timestamp': datetime.now().isoformat()
            }), 409

    except Exception as e:
        print(f"❌ 用户注册失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/auth/login', methods=['POST'])
def login():
    """
    用户登录
    
    请求体格式:
    {
        "username": "用户名",
        "password": "密码"
    }
    """
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是 JSON 格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({
                'success': False,
                'error': '用户名和密码不能为空',
                'timestamp': datetime.now().isoformat()
            }), 400

        user = db_manager.login_user(username, password)

        if user:
            return jsonify({
                'success': True,
                'message': '登录成功',
                'user': user,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': '用户名或密码错误',
                'timestamp': datetime.now().isoformat()
            }), 401

    except Exception as e:
        print(f"❌ 用户登录失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ================ 对话会话管理 API 端点 ================
@app.route('/conversation/sessions', methods=['GET'])
def get_conversation_sessions():
    """获取用户的所有会话"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        user_id = request.args.get('user_id', type=int)
        if not user_id:
            return jsonify({
                'success': False,
                'error': '请提供user_id参数',
                'timestamp': datetime.now().isoformat()
            }), 400

        sessions = db_manager.get_user_conversation_sessions(user_id)
        return jsonify({
            'success': True,
            'sessions': sessions,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 获取会话列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/conversation/sessions', methods=['POST'])
def create_conversation_session():
    """创建新会话"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是JSON格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json()
        user_id = data.get('user_id')
        title = data.get('title', '新对话')

        if not user_id:
            return jsonify({
                'success': False,
                'error': '请提供user_id',
                'timestamp': datetime.now().isoformat()
            }), 400

        session = db_manager.create_conversation_session(user_id, title)
        if session:
            return jsonify({
                'success': True,
                'session': session,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': '创建会话失败',
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        print(f"❌ 创建会话失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/conversation/sessions/<session_id>', methods=['GET'])
def get_conversation_session(session_id):
    """获取单个会话"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        session = db_manager.get_conversation_session(session_id)
        if session:
            return jsonify({
                'success': True,
                'session': session,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': '会话不存在',
                'timestamp': datetime.now().isoformat()
            }), 404

    except Exception as e:
        print(f"❌ 获取会话失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/conversation/sessions/<session_id>', methods=['PUT'])
def update_conversation_session(session_id):
    """更新会话"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求必须是JSON格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json()
        title = data.get('title')

        success = db_manager.update_conversation_session(session_id, title)
        if success:
            session = db_manager.get_conversation_session(session_id)
            return jsonify({
                'success': True,
                'session': session,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': '更新会话失败',
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        print(f"❌ 更新会话失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/conversation/sessions/<session_id>', methods=['DELETE'])
def delete_conversation_session(session_id):
    """删除会话"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        success = db_manager.delete_conversation_session(session_id)
        if success:
            return jsonify({
                'success': True,
                'message': '会话已删除',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': '删除会话失败',
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        print(f"❌ 删除会话失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/conversation/sessions/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """获取会话的对话历史"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        user_id = request.args.get('user_id', type=int)
        limit = request.args.get('limit', 50, type=int)

        if not user_id:
            return jsonify({
                'success': False,
                'error': '请提供user_id参数',
                'timestamp': datetime.now().isoformat()
            }), 400

        history = db_manager.get_conversation_history(user_id, session_id, limit)
        return jsonify({
            'success': True,
            'history': history,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 获取对话历史失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/conversation/sessions/<session_id>/summaries', methods=['GET'])
def get_session_summaries(session_id):
    """获取会话的总结"""
    try:
        if not DATABASE_AVAILABLE or db_manager is None:
            return jsonify({
                'success': False,
                'error': '数据库不可用',
                'timestamp': datetime.now().isoformat()
            }), 503

        limit = request.args.get('limit', 2, type=int)
        summaries = db_manager.get_session_summaries(session_id, limit)
        return jsonify({
            'success': True,
            'summaries': summaries,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ 获取会话总结失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# ================ 启动服务器 ================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🔄 正在初始化系统组件...")
    print("=" * 70)

    # 初始化所有组件
    model_initialized = initialize_model()
    health_initialized = initialize_health_advisor()

    # 初始化数据库管理器（在摄像头监测器之前）
    db_initialized = False
    if DATABASE_AVAILABLE:
        try:
            db_manager = get_db_manager()
            db_initialized = True
            print("✅ 数据库管理器初始化成功")
        except Exception as e:
            print(f"❌ 数据库管理器初始化失败: {e}")

    # 初始化摄像头监测器
    camera_initialized = initialize_camera_monitor()

    # 初始化 LangGraph Agent
    langgraph_agent_initialized = False
    if LANGGRAPH_AVAILABLE:
        try:
            langgraph_agent = get_langgraph_agent()
            langgraph_agent_initialized = True
            print("✅ LangGraph Agent 初始化成功")
        except Exception as e:
            print(f"❌ LangGraph Agent 初始化失败: {e}")

    print("\n" + "=" * 70)
    print("✅ 初始化完成")
    print("=" * 70)

    print(f"📊 初始化状态:")
    print(f"  表情识别模型: {'✅ 已加载' if model_initialized else '❌ 未加载'}")
    print(f"  健康建议模块: {'✅ 已加载' if health_initialized else '❌ 未加载'}")
    print(f"  摄像头监测器: {'✅ 已加载' if camera_initialized else '❌ 未加载'}")
    print(f"  阿里云大模型: {'✅ 可用' if OPENAI_AVAILABLE else '❌ 不可用'}")
    print(f"  数据库: {'✅ 已加载' if db_initialized else '❌ 未加载'}")
    print(f"  LangGraph Agent: {'✅ 已加载' if langgraph_agent_initialized else '❌ 未加载'}")

    print("\n" + "=" * 70)
    print("🌐 API 服务器启动中...")
    print("=" * 70)

    print(f"📌 访问地址: http://0.0.0.0:7860")
    print(f"📌 健康检查: http://0.0.0.0:7860/health")
    print(f"📌 情绪列表: http://0.0.0.0:7860/emotions")
    print(f"📌 LangGraph Agent 对话: http://0.0.0.0:7860/langgraph/chat")

    print(f"\n🔧 测试命令:")
    print(f"  curl http://0.0.0.0:7860/health")
    print(f"  curl http://0.0.0.0:7860/emotions")
    print(
        f"  curl -X POST http://0.0.0.0:7860/comprehensive_analysis -H 'Content-Type: application/json' -d '{{\"analysis_type\":\"health_advice\",\"days\":7}}'")
    print(
        f"  curl -X POST http://0.0.0.0:7860/langgraph/chat -H 'Content-Type: application/json' -d '{{\"message\":\"你好\"}}'")

    print(f"\n⚠️  注意:")
    print(f"  阿里云大模型API Key: {'已设置' if os.getenv('DASHSCOPE_API_KEY') else '未设置（需在环境变量中配置 DASHSCOPE_API_KEY）'}")
    print(f"  综合情绪分析端点: POST /comprehensive_analysis")
    print(f"  LangGraph Agent 端点: POST /langgraph/chat")
    print("=" * 70)

    # 启动Flask服务器
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)