# 🧠 EmoCare - 智能情绪识别与健康管理系统

> 仓库已重组：后端代码已移动到 `backend/`，前端示例已移动到 `frontend/`。如需旧路径参考，请查看 `PROJECT_STRUCTURE.md`。

> **基于深度学习与大模型的心理健康数字助手** > **西安交通大学 大学生创新创业训练项目**


## 📖 项目简介 (Introduction)

**EmoCare** 是一套结合了**计算机视觉 (CV)** 与 **生成式人工智能 (GenAI)** 的情绪健康管理系统。

传统的心理评估往往依赖滞后的问卷调查，而本系统通过摄像头实时捕捉面部微表情，利用 **ResNet** 深度神经网络精准识别 6 种基本情绪（快乐、悲伤、愤怒、恐惧、厌恶、惊讶）。系统不仅能“看懂”你的表情，还能通过内置的**专家规则库**与**阿里云百炼大模型**，为您提供即时、温暖的心理调节建议。

### ✨ 核心功能
* **📷 多模态识别**：支持 **实时摄像头监测** 与 **静态图片上传** 分析。
* **📊 六维情绪量化**：精准计算并展示情绪的概率分布（雷达图），捕捉复杂混合情绪。
* **💡 智能健康顾问**：
    * **规则引擎**：基于心理学专家的“急救包”建议（如愤怒时的深呼吸引导）。
    * **LLM 深度分析**：集成阿里云大模型，生成富有共情力的综合心理分析报告。
* **🗄️ MySQL 数据存储**：支持将用户情绪监测数据和对话历史存储到 MySQL 数据库。
* **🤖 LangGraph 智能对话系统**：
    * 支持创建、选择、删除对话会话
    * 每轮对话自动生成总结，结合前两轮总结和本轮内容
    * 对前25轮对话保持记忆，基于总结进行记忆
    * 异步处理，快速响应，后台继续保存和总结
    * 优化的大模型调用策略，减少等待时间
* **⚡ 极速部署**：提供一键启动脚本，自动配置虚拟环境与依赖。

---

## 🛠️ 技术架构 (Tech Stack)

* **前端交互**：HTML5, JavaScript (Fetch API), Chart.js (数据可视化)
* **后端服务**：Python Flask (RESTful API)
* **核心算法**：
    * **模型**：ResNet50 / ResNet18 (PyTorch)
  * **预处理**：Facenet MTCNN 人脸检测（失败时回退 OpenCV Haar）
* **大模型集成**：Alibaba DashScope (通义千问 / OpenAI Compatible SDK)

---

## 📂 目录结构 (Directory Structure)

```text
EmoCare/
├── backend/
│   ├── api/                # 后端 API 代码
│   │   ├── api_server.py   # Flask 服务器入口
│   │   ├── emotion_agent.py # 多模态智能 Agent (基于 LangChain)
│   │   ├── api_client.py   # API 客户端
│   │   ├── camera_monitor.py # 摄像头监测模块
│   │   └── data/           # 监测数据存储
│   ├── models/             # 模型定义与处理逻辑
│   │   ├── emotion_model.py # PyTorch ResNet 模型结构
│   │   └── health_advisor.py # 建议生成与 LLM 接口逻辑
│   ├── docs/               # 项目文档
│   └── requirements.txt    # 项目依赖列表
├── frontend/
│   └── examples/           # 前端界面示例
│       └── emotion_ui.html # 用户交互界面
├── data/                   # 数据存储（日志/临时文件）
├── facenet/                # 人脸识别接口
├── best_model.pth          # 训练好的模型权重文件
├── advice_rules.json       # 心理健康建议规则库
├── start.bat               # ✅ Windows 一键启动脚本
└── README.md               # 项目说明文档



🚀 快速开始 (Quick Start)
### 1. 环境准备
- **操作系统**：Windows 10/11
- **Python 版本**：建议 Python 3.10 - 3.13
- **摄像头**：用于实时监测功能（可选）
### 2. 配置环境变量

复制项目根目录下的 `.env.example` 为 `.env`，然后修改以下配置：

```env
# MySQL 数据库配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的MySQL密码
MYSQL_DATABASE=emocare

# 阿里云 API Key (通义千问)
DASHSCOPE_API_KEY=sk-你的阿里云API密钥
```

### 3. 创建数据库和表

有两种方式可以初始化数据库：

#### 方式一：使用 MySQL Workbench

1. 打开 MySQL Workbench
2. 连接到你的 MySQL 服务器
3. 打开并执行 `backend/database/schema.sql` 文件中的 SQL 语句

#### 方式二：使用初始化脚本

```bash
conda activate emotiondetection
python backend/database/init_db.py
```

这个脚本会自动：
- 创建 `emocare` 数据库
- 创建所有必需的表
- 迁移现有的 JSON 监测数据到 MySQL
### 4.创建默认用户
运行以下命令创建默认用户：
```bash
python backend/database/create_default_user.py
```
### 5. 配置人脸识别接口环境
1. 从 https://github.com/timesler/facenet-pytorch fork人脸识别项目  
2. 将fork的项目facenet克隆到该项目的根目录
3. 配置facenet`python -m pip install -e .\facenet`

### 6. Conda 环境配置（若使用conda+cuda参考此启动方法）

(1) 创建 conda 环境
```bash
conda create -n emotiondetection python=3.10
```

(2) 激活 conda 环境
```bash
conda activate emotiondetection
```

(3) 安装依赖
```bash
pip install -r backend/requirements.txt
python -m pip install -e .\facenet
```

（4）启动项目
方式一：一键启动（推荐）
双击项目根目录下的 `start.bat`，或在命令行中运行：
```bash
start.bat
```

方式二：手动启动
如果 `start.bat` 无法正常工作，可以手动分别启动两个服务器：

**终端 1 - 启动 API 服务器：**
```bash
conda activate emotiondetection
python backend/api/api_server.py
```

**终端 2 - 启动 HTTP 文件服务器：**
```bash
conda activate emotiondetection
python -m http.server 8000
```

### 7. venv 环境配置（若使用venv环境参考此启动方法）

如果您不想使用 conda 环境，可以使用 .venv 虚拟环境：

#### 方式一：一键启动（推荐）
双击项目根目录下的 `start_venv.bat`，或在命令行中运行：
```bash
start_venv.bat
```

这个脚本会自动：
- 创建 .venv 虚拟环境（如果不存在）
- 自动升级 pip 到最新版本
- 安装 Python 3.13 兼容的依赖（flask==2.3.3, werkzeug==2.3.7, flask-cors>=4.0.0）
- 从 backend/requirements.txt 安装其他项目依赖
- 启动 API 服务器（端口 7860）
- 启动 HTTP 文件服务器（端口 8000）
- 自动打开浏览器

#### 方式二：手动启动
如果 `start_venv.bat` 无法正常工作，可以手动分别启动两个服务器：

**终端 1 - 启动 API 服务器：**
```bash
# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 升级 pip
python -m pip install --upgrade pip

# 安装 Python 3.13 兼容的依赖
pip install flask==2.3.3 werkzeug==2.3.7
pip install flask-cors>=4.0.0 --upgrade

# 安装其他依赖
pip install -r backend\requirements.txt --ignore-installed flask flask-cors werkzeug

# 启动 API 服务器
python backend\api\api_server.py
```

**终端 2 - 启动 HTTP 文件服务器：**
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 启动 HTTP 服务器
python -m http.server 8000
```

### 8. 访问界面
启动成功后，在浏览器中访问：
👉 **http://localhost:8000/frontend/examples/emotion_ui.html**

**注意**：请务必保留 URL 中的 `/frontend/examples/` 路径，否则会导致页面资源加载失败。

⚙️ 配置说明 (Configuration)
启用 AI 大模型分析功能
系统默认使用本地规则库生成建议。如果您希望启用基于 阿里云通义千问 的深度分析功能，请按以下步骤配置 API Key：

打开 start_api.bat 文件（右键 -> 编辑）。

在文件顶部的 setlocal 下方添加一行：
set DASHSCOPE_API_KEY=sk-您的阿里云API密钥

保存并重启脚本。

⚠️ 常见问题 (FAQ)
Q: 打开网页显示 404 Not Found？ A: 请确认您的访问地址是 http://localhost:8000/frontend/examples/emotion_ui.html。如果不包含 frontend/examples，服务器将无法找到前端文件。

Q: 启动脚本闪退？ A: 请尝试右键点击 start_api.bat 选择“以管理员身份运行”。如果问题依旧，请在文件夹地址栏输入 cmd，然后手动运行脚本查看具体报错信息。

Q: 摄像头无法开启？ A: 请确保浏览器已获得摄像头权限（通常在地址栏左侧有个锁图标或摄像机图标，点击允许即可）。


---

## 🗄️ MySQL 数据库配置 (MySQL Configuration)

### 1. 准备 MySQL 数据库

确保你已安装并启动了 MySQL 服务器（推荐使用 MySQL Workbench 进行管理）。

### 2. 配置环境变量

复制项目根目录下的 `.env.example` 为 `.env`，然后修改以下配置：

```env
# MySQL 数据库配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的MySQL密码
MYSQL_DATABASE=emocare

# 阿里云 API Key (通义千问)
DASHSCOPE_API_KEY=sk-你的阿里云API密钥
```

### 3. 创建数据库和表

有两种方式可以初始化数据库：

#### 方式一：使用 MySQL Workbench

1. 打开 MySQL Workbench
2. 连接到你的 MySQL 服务器
3. 打开并执行 `backend/database/schema.sql` 文件中的 SQL 语句

#### 方式二：使用初始化脚本

```bash
conda activate emotiondetection
python backend/database/init_db.py
```

这个脚本会自动：
- 创建 `emocare` 数据库
- 创建所有必需的表
- 迁移现有的 JSON 监测数据到 MySQL

### 4. 数据库表结构说明

| 表名 | 说明 |
|------|------|
| `users` | 用户信息表 |
| `conversation_sessions` | 对话会话表 |
| `session_summaries` | 会话总结表 |
| `emotion_records` | 情绪监测记录表 |
| `user_personality_analysis` | 用户性格分析表 |
| `conversation_history` | 对话历史表 |
| `system_prompt_cache` | 系统提示词缓存表 |



## 🤖 LangGraph 智能对话系统 (LangGraph Agent)

### 工作原理

LangGraph Agent 提供了一个智能的个性化对话系统，工作流程如下：

1. **优化的快速响应**：
   - 异步处理：生成AI回复后立即返回给前端
   - 后台继续保存对话和生成总结，不阻塞用户体验
   - 简化流程：跳过性格和情绪分析，专注于快速对话
   - 减少大模型调用：每10轮对话才生成一次总结

2. **对话会话管理**：
   - 支持创建、选择、删除对话会话
   - 删除会话时级联删除所有相关数据（历史、总结等）
   - 每轮对话自动生成总结
   - 对前25轮对话保持记忆

### API 端点

#### 对话会话管理 API

##### 获取用户的所有会话
```bash
GET /conversation/sessions?user_id=1
```

##### 创建新会话
```bash
POST /conversation/sessions
Content-Type: application/json

{
  "user_id": 1,
  "title": "新对话"
}
```

##### 获取单个会话
```bash
GET /conversation/sessions/{session_id}
```

##### 更新会话
```bash
PUT /conversation/sessions/{session_id}
Content-Type: application/json

{
  "title": "更新后的标题"
}
```

##### 删除会话（级联删除所有相关数据）
```bash
DELETE /conversation/sessions/{session_id}
```

##### 获取会话的对话历史
```bash
GET /conversation/sessions/{session_id}/history?user_id=1&limit=50
```

##### 获取会话的总结
```bash
GET /conversation/sessions/{session_id}/summaries?limit=2
```

#### 与 LangGraph Agent 对话
```bash
POST /langgraph/chat
Content-Type: application/json

{
  "message": "你好，我最近心情怎么样？",
  "user_id": 1,
  "session_id": "my_session_123"
}
```

#### 获取 LangGraph Agent 状态
```bash
GET /langgraph/status
```

#### 保存情绪记录到数据库
```bash
POST /database/save_emotion
Content-Type: application/json

{
  "emotion_data": {
    "emotion": "happy",
    "emotion_zh": "快乐",
    "confidence": 0.85,
    "probabilities": {"happy": 0.85, "neutral": 0.15}
  },
  "user_id": 1,
  "session_id": "monitor_session"
}
```


---

## 📂 更新后的目录结构 (Updated Directory Structure)

```text
EmoCare/
├── backend/
│   ├── api/                # 后端 API 代码
│   │   ├── api_server.py   # Flask 服务器入口
│   │   ├── emotion_agent.py # 多模态智能 Agent (基于 LangChain)
│   │   ├── langgraph_agent.py # LangGraph 智能对话系统（已优化速度）
│   │   ├── api_client.py   # API 客户端
│   │   ├── camera_monitor.py # 摄像头监测模块
│   │   └── data/           # 监测数据存储
│   ├── database/           # 数据库模块
│   │   ├── db_manager.py   # 数据库管理模块
│   │   ├── init_db.py      # 数据库初始化脚本
│   │   ├── schema.sql      # 数据库表结构（最新）
│   │   └── update_foreign_keys.sql # 更新外键约束脚本
│   ├── models/             # 模型定义与处理逻辑
│   │   ├── emotion_model.py # PyTorch ResNet 模型结构
│   │   └── health_advisor.py # 建议生成与 LLM 接口逻辑
│   ├── docs/               # 项目文档
│   └── requirements.txt    # 项目依赖列表
├── frontend/
│   └── examples/           # 前端界面示例
│       └── emotion_ui.html # 用户交互界面（含会话管理）
├── data/                   # 数据存储（日志/临时文件）
├── best_model.pth          # 训练好的模型权重文件
├── advice_rules.json       # 心理健康建议规则库
├── .env.example            # 环境变量配置示例
├── start.bat               # ✅ Windows 一键启动脚本
└── README.md               # 项目说明文档
```