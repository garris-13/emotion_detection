"""
基于 LangGraph 的智能对话系统
实现用户数据分析和个性化对话
"""
import os
import sys
import json
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Annotated

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# LangChain/LangGraph 导入
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️ LangGraph 导入失败: {e}")

# 导入数据库管理器
from backend.database import get_db_manager


# 定义状态类型
class AgentState(TypedDict):
    user_id: int
    session_id: str
    user_message: str
    personality_summary: Optional[str]
    recent_emotion_summary: Optional[str]
    system_prompt: Optional[str]
    conversation_history: List[Dict]
    response: Optional[str]
    step: str


class LangGraphEmotionAgent:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.llm = None
        self.graph = None
        self.checkpointer = None
        self.db = None
        
        if LANGGRAPH_AVAILABLE and self.api_key:
            self._init_llm()
            self._init_db()
            self._build_graph()
        else:
            print("⚠️ Agent 处于降级模式：LangGraph 或 API Key 不可用")

    def _init_llm(self):
        """初始化 LLM"""
        try:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model="qwen3.5-plus",
                temperature=0.7,
                max_tokens=2000
            )
            print("✅ LLM 初始化成功")
        except Exception as e:
            print(f"❌ LLM 初始化失败: {e}")

    def _init_db(self):
        """初始化数据库连接"""
        try:
            self.db = get_db_manager()
            print("✅ 数据库连接成功")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")

    def _build_graph(self):
        """构建 LangGraph 状态图（完整版 - 包含性格和情绪分析）"""
        # 暂时禁用 SQLite 检查点存储器（Windows 兼容性问题）
        self.checkpointer = None  # 暂时禁用检查点

        # 构建状态图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("check_cache", self._check_cache_node)
        workflow.add_node("analyze_personality", self._analyze_personality_node)
        workflow.add_node("analyze_recent_emotion", self._analyze_recent_emotion_node)
        workflow.add_node("build_system_prompt", self._build_system_prompt_node)
        workflow.add_node("save_cache", self._save_cache_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # 设置入口点
        workflow.set_entry_point("check_cache")
        
        # 条件边：决定是否使用缓存
        workflow.add_conditional_edges(
            "check_cache",
            self._should_use_cache,
            {
                "use_cache": "generate_response",
                "analyze": "analyze_personality"
            }
        )
        
        # 分析流程
        workflow.add_edge("analyze_personality", "analyze_recent_emotion")
        workflow.add_edge("analyze_recent_emotion", "build_system_prompt")
        workflow.add_edge("build_system_prompt", "save_cache")
        workflow.add_edge("save_cache", "generate_response")
        workflow.add_edge("generate_response", END)

        # 编译图（不使用检查点）
        self.graph = workflow.compile()
        print("✅ LangGraph 构建成功（完整版 - 包含性格和情绪分析）")
    
    def _generate_simple_response_node(self, state: AgentState) -> AgentState:
        """简化版回复生成节点 - 不做性格和情绪分析"""
        print("💬 快速生成回复...")
        
        user_message = state["user_message"]
        conversation_history = state.get("conversation_history", [])
        session_id = state["session_id"]
        user_id = state["user_id"]
        
        # 简化的系统提示词
        simple_system_prompt = """你是一位温暖、贴心、富有同理心的心理健康陪伴者，名叫"EmoCare"。

你的核心定位：
- 你是用户的知心朋友和心理健康伙伴
- 用温暖、关怀的语气与用户对话
- 认真倾听用户的心声，给予情感支持

你的沟通风格：
- 温暖亲切，像朋友一样聊天
- 富有同理心，能够理解用户的感受
- 善于倾听，不急于给出建议
- 多用鼓励和肯定的语言
- 避免说教，用轻松自然的方式交流

注意事项：
- 如果用户情绪低落，先给予温暖的安慰和陪伴
- 不要轻易评判用户的情绪
- 尊重用户的隐私和选择
- 如发现严重心理问题，温柔地建议寻求专业帮助
- 始终保持耐心和温暖，让用户感到被理解和被关心

请用中文回复，回复简洁友好，不超过300字。"""
        
        # 获取会话的总结（前2轮）用于记忆
        session_summaries = []
        if self.db:
            session_summaries = self.db.get_session_summaries(session_id, limit=2)
        
        # 构建对话提示
        messages = [
            SystemMessage(content=simple_system_prompt)
        ]
        
        # 添加历史对话（前25轮）
        for msg in conversation_history[-25:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前用户消息
        messages.append(HumanMessage(content=user_message))
        
        try:
            response = self.llm.invoke(messages)
            return {**state, "response": response.content, "personality_summary": None, "recent_emotion_summary": None, "system_prompt": simple_system_prompt, "step": "simple"}
        except Exception as e:
            print(f"❌ 生成回复失败: {e}")
            return {**state, "response": "抱歉，我遇到了一些问题。让我们换个话题聊聊吧？", "personality_summary": None, "recent_emotion_summary": None, "system_prompt": simple_system_prompt, "step": "simple"}

    def _check_cache_node(self, state: AgentState) -> AgentState:
        """检查缓存节点"""
        print("🔍 检查系统提示词缓存...")
        
        user_id = state["user_id"]
        session_id = state["session_id"]
        
        # 从数据库检查缓存
        cache = None
        if self.db:
            cache = self.db.get_system_prompt_cache(user_id, session_id)
        
        if cache:
            print("✅ 使用缓存的系统提示词")
            return {
                **state,
                "personality_summary": cache["personality_summary"],
                "recent_emotion_summary": cache["recent_emotion_summary"],
                "system_prompt": cache["full_system_prompt"],
                "step": "use_cache"
            }
        else:
            print("📊 无缓存，需要分析用户数据")
            return {**state, "step": "analyze"}

    def _should_use_cache(self, state: AgentState) -> str:
        """决定是否使用缓存"""
        return "use_cache" if state["step"] == "use_cache" else "analyze"

    def _analyze_personality_node(self, state: AgentState) -> AgentState:
        """分析用户性格（半年左右）"""
        print("🧠 分析用户性格（半年左右）...")
        
        user_id = state["user_id"]
        
        # 从数据库获取半年的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6个月
        
        records = []
        if self.db:
            records = self.db.get_emotion_records(user_id, start_date=start_date, end_date=end_date)
        
        if not records:
            personality_summary = "暂无足够的历史数据来分析您的性格特征。"
        else:
            # 调用 LLM 分析性格
            personality_summary = self._analyze_personality_with_llm(records)
            
            # 计算情绪分布用于存储
            emotion_counts = {}
            for record in records:
                emo = record['emotion']
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            
            # 计算稳定性分数（简单版：情绪多样性的反面）
            total_records = len(records)
            dominant_count = max(emotion_counts.values()) if emotion_counts else 0
            stability_score = (dominant_count / total_records * 100) if total_records > 0 else 50
            
            # 保存到数据库
            if self.db:
                try:
                    self.db.save_personality_analysis(
                        user_id=user_id,
                        analysis_period='6months',
                        start_date=start_date,
                        end_date=end_date,
                        personality_summary=personality_summary,
                        emotion_patterns=emotion_counts,
                        dominant_emotions=emotion_counts,
                        stability_score=stability_score
                    )
                    print("💾 性格分析已保存到数据库")
                except Exception as e:
                    print(f"⚠️ 保存性格分析失败: {e}")
        
        return {**state, "personality_summary": personality_summary}

    def _analyze_personality_with_llm(self, records: List[Dict]) -> str:
        """使用 LLM 分析用户性格"""
        emotion_zh_map = {
            'surprised': '惊讶', 'fear': '恐惧', 'disgust': '厌恶',
            'happy': '快乐', 'sad': '悲伤', 'anger': '愤怒', 'neutral': '平静'
        }
        
        # 准备数据摘要
        data_summary = []
        for record in records[-50:]:  # 最多用50条
            emo = emotion_zh_map.get(record['emotion'], record['emotion'])
            data_summary.append(f"{record['recorded_at']}: {emo} (置信度: {record['confidence']:.2f})")
        
        prompt = f"""你是一位专业的性格分析师。请根据以下用户的情绪历史数据，分析用户的性格特征。

情绪历史数据：
{chr(10).join(data_summary[:50])}

请从以下角度分析用户性格：
1. 情绪稳定性
2. 主要情绪倾向
3. 应对压力的方式
4. 社交倾向
5. 其他可观察的性格特征

请用温暖、关怀的语言总结，不超过300字。"""

        messages = [
            SystemMessage(content="你是一位专业且温暖的性格分析师。"),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"❌ 性格分析失败: {e}")
            return "根据您的情绪数据，您是一位情感丰富的人。"

    def _analyze_recent_emotion_node(self, state: AgentState) -> AgentState:
        """分析用户最近情绪（1个月左右 + 最近3次）"""
        print("📈 分析用户最近情绪（1个月左右 + 最近3次）...")
        
        user_id = state["user_id"]
        
        # 从数据库获取最近1个月的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 1个月
        
        records_month = []
        records_recent_3 = []
        if self.db:
            records_month = self.db.get_emotion_records(user_id, start_date=start_date, end_date=end_date)
            # 获取最近3次记录（不限制时间）
            all_records = self.db.get_emotion_records(user_id)
            records_recent_3 = all_records[:3] if all_records else []
        
        if not records_month and not records_recent_3:
            recent_emotion_summary = "暂无最近的情绪监测数据。"
        else:
            # 调用 LLM 分析最近情绪
            recent_emotion_summary = self._analyze_recent_emotion_with_llm(records_month, records_recent_3)
            
            # 计算情绪分布用于存储
            all_records_for_save = records_month + records_recent_3
            emotion_counts = {}
            for record in all_records_for_save:
                emo = record['emotion']
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            
            # 计算稳定性分数
            total_records = len(all_records_for_save)
            dominant_count = max(emotion_counts.values()) if emotion_counts else 0
            stability_score = (dominant_count / total_records * 100) if total_records > 0 else 50
            
            # 保存到数据库（使用 1month 周期）
            if self.db:
                try:
                    self.db.save_personality_analysis(
                        user_id=user_id,
                        analysis_period='1month',
                        start_date=start_date,
                        end_date=end_date,
                        personality_summary=recent_emotion_summary,
                        emotion_patterns=emotion_counts,
                        dominant_emotions=emotion_counts,
                        stability_score=stability_score
                    )
                    print("💾 最近情绪分析已保存到数据库")
                except Exception as e:
                    print(f"⚠️ 保存最近情绪分析失败: {e}")
        
        return {**state, "recent_emotion_summary": recent_emotion_summary}

    def _analyze_recent_emotion_with_llm(self, records_month: List[Dict], records_recent_3: List[Dict]) -> str:
        """使用 LLM 分析最近情绪（1个月 + 最近3次）"""
        emotion_zh_map = {
            'surprised': '惊讶', 'fear': '恐惧', 'disgust': '厌恶',
            'happy': '快乐', 'sad': '悲伤', 'anger': '愤怒', 'neutral': '平静'
        }
        
        # 统计1个月的情绪分布
        emotion_counts = {}
        for record in records_month:
            emo = emotion_zh_map.get(record['emotion'], record['emotion'])
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        
        # 准备1个月数据摘要
        data_summary_month = []
        for record in records_month[-20:]:  # 最近20条
            emo = emotion_zh_map.get(record['emotion'], record['emotion'])
            data_summary_month.append(f"{record['recorded_at'].strftime('%m-%d %H:%M')}: {emo}")
        
        # 准备最近3次数据
        data_summary_recent = []
        for record in records_recent_3:
            emo = emotion_zh_map.get(record['emotion'], record['emotion'])
            data_summary_recent.append(f"{record['recorded_at'].strftime('%m-%d %H:%M')}: {emo} (置信度: {record['confidence']:.2f})")
        
        dist_str = ', '.join([f"{k}: {v}次" for k, v in emotion_counts.items()]) if emotion_counts else "无数据"
        
        prompt = f"""你是一位温暖的情绪健康顾问。请根据以下用户的情绪数据，分析用户的情绪状态。

【近1个月情绪数据】（共{len(records_month)}条）
{chr(10).join(data_summary_month) if data_summary_month else "无数据"}

情绪分布：{dist_str}

【最近3次情绪】
{chr(10).join(data_summary_recent) if data_summary_recent else "无数据"}

请从以下角度分析：
1. 当前情绪状态（基于最近3次）
2. 近1个月情绪变化趋势
3. 需要关注的点
4. 简单的建议

请用温暖、关怀的语言总结，不超过250字。"""

        messages = [
            SystemMessage(content="你是一位温暖且专业的情绪健康顾问。"),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"❌ 最近情绪分析失败: {e}")
            return "您最近的情绪状态看起来比较平稳。"

    def _build_system_prompt_node(self, state: AgentState) -> AgentState:
        """构建系统提示词"""
        print("📝 构建系统提示词...")
        
        personality_summary = state["personality_summary"]
        recent_emotion_summary = state["recent_emotion_summary"]
        
        system_prompt = f"""你是一位温暖、贴心、富有同理心的心理健康陪伴者，名叫"EmoCare"。

【用户性格分析】
{personality_summary}

【最近情绪状态】
{recent_emotion_summary}

你的核心定位：
- 你是用户的知心朋友和心理健康伙伴
- 用温暖、关怀的语气与用户对话
- 认真倾听用户的心声，给予情感支持
- 基于用户的性格特征和情绪状态，提供个性化的关怀和建议

你的沟通风格：
- 温暖亲切，像朋友一样聊天
- 富有同理心，能够理解用户的感受
- 善于倾听，不急于给出建议
- 多用鼓励和肯定的语言
- 避免说教，用轻松自然的方式交流

注意事项：
- 如果用户情绪低落，先给予温暖的安慰和陪伴
- 不要轻易评判用户的情绪
- 尊重用户的隐私和选择
- 如发现严重心理问题，温柔地建议寻求专业帮助
- 始终保持耐心和温暖，让用户感到被理解和被关心

请根据以上用户分析，与用户进行个性化的对话。"""
        
        return {**state, "system_prompt": system_prompt}

    def _save_cache_node(self, state: AgentState) -> AgentState:
        """保存缓存"""
        print("💾 保存系统提示词缓存...")
        
        user_id = state["user_id"]
        session_id = state["session_id"]
        
        if self.db:
            self.db.save_system_prompt_cache(
                user_id=user_id,
                session_id=session_id,
                personality_summary=state["personality_summary"],
                recent_emotion_summary=state["recent_emotion_summary"],
                full_system_prompt=state["system_prompt"],
                expires_hours=24
            )
        
        return state

    def _generate_response_node(self, state: AgentState) -> AgentState:
        """生成回复"""
        print("💬 生成回复...")
        
        user_message = state["user_message"]
        system_prompt = state["system_prompt"]
        conversation_history = state.get("conversation_history", [])
        session_id = state["session_id"]
        user_id = state["user_id"]
        
        # 获取会话的总结（前2轮）用于记忆
        session_summaries = []
        if self.db:
            session_summaries = self.db.get_session_summaries(session_id, limit=2)
        
        # 构建系统提示词，添加总结记忆
        enhanced_system_prompt = system_prompt
        if session_summaries:
            summary_text = "\n".join([f"- {s['summary_text']}" for s in session_summaries])
            enhanced_system_prompt += f"\n\n【对话历史总结】\n{summary_text}\n\n请基于以上总结和当前对话，提供连贯的回复。"
        
        # 构建对话提示
        messages = [
            SystemMessage(content=enhanced_system_prompt)
        ]
        
        # 添加历史对话（前25轮）
        for msg in conversation_history[-25:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前用户消息
        messages.append(HumanMessage(content=user_message))
        
        try:
            response = self.llm.invoke(messages)
            return {**state, "response": response.content}
        except Exception as e:
            print(f"❌ 生成回复失败: {e}")
            return {**state, "response": "抱歉，我遇到了一些问题。让我们换个话题聊聊吧？"}
    
    def _generate_conversation_summary(self, user_id: int, session_id: str, 
                                       current_user_msg: str, current_assistant_msg: str) -> Optional[str]:
        """生成对话总结（优化版 - 简洁快速）"""
        print("📝 生成对话总结...")
        
        # 简化提示词，减少token消耗和生成时间
        summary_prompt = f"""请用一句话（不超过50字）总结以下对话：
用户：{current_user_msg[:100]}
助手：{current_assistant_msg[:100]}"""
        
        messages = [
            SystemMessage(content="你是一位简洁的对话总结助手。"),
            HumanMessage(content=summary_prompt)
        ]
        
        try:
            # 使用更快速的模型
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"⚠️ 生成对话总结失败: {e}")
            return None

    def chat(self, user_message: str, user_id: int = 1, session_id: str = None, 
             create_session: bool = True) -> Dict[str, Any]:
        """与 Agent 对话（完整版 - 包含性格和情绪分析）"""
        if not self.graph:
            return {
                "success": False,
                "error": "LangGraph 未初始化",
                "response": "抱歉，功能暂不可用。"
            }
        
        # 检查会话是否存在，如果不存在则创建
        is_new_session = False
        if self.db and session_id:
            existing_session = self.db.get_conversation_session(session_id)
            if not existing_session and create_session:
                # 创建新会话
                title = user_message[:20] if len(user_message) > 20 else user_message
                new_session = self.db.create_conversation_session(user_id, title)
                if new_session:
                    session_id = new_session['id']
                    is_new_session = True
        
        if not session_id:
            # 自动创建会话
            if self.db and create_session:
                title = user_message[:20] if len(user_message) > 20 else user_message
                new_session = self.db.create_conversation_session(user_id, title)
                if new_session:
                    session_id = new_session['id']
                    is_new_session = True
            if not session_id:
                session_id = str(uuid.uuid4())
                is_new_session = True
        
        # 获取对话历史
        conversation_history = []
        message_count = 0
        if self.db:
            conversation_history = self.db.get_conversation_history(user_id, session_id)
            message_count = len(conversation_history)
        
        # 如果是新会话，清除缓存，强制重新分析
        if is_new_session and self.db:
            try:
                # 删除该会话的缓存
                delete_query = "DELETE FROM system_prompt_cache WHERE user_id = %s AND session_id = %s"
                self.db._execute_query(delete_query, (user_id, session_id), commit=True)
                print(f"🗑️ 已清除新会话的缓存，将重新分析用户数据")
            except Exception as e:
                print(f"⚠️ 清除缓存失败: {e}")
        
        # 初始状态 - 新会话强制分析
        initial_state: AgentState = {
            "user_id": user_id,
            "session_id": session_id,
            "user_message": user_message,
            "personality_summary": None,
            "recent_emotion_summary": None,
            "system_prompt": None,
            "conversation_history": conversation_history,
            "response": None,
            "step": "analyze" if is_new_session else "check_cache"
        }
        
        try:
            # 执行图（不使用检查点）
            final_state = self.graph.invoke(initial_state)
            assistant_response = final_state["response"]
            
            # 立即返回响应，后台异步处理保存和总结
            result = {
                "success": True,
                "response": assistant_response,
                "session_id": session_id,
                "personality_analyzed": False,
                "used_cache": False,
                "context_summary": None
            }
            
            # 后台异步处理保存和总结
            if self.db:
                # 每5轮生成一次总结，减少大模型调用
                should_generate_summary = (message_count + 2) % 10 == 0  # 每10轮消息生成一次总结（用户+助手算2轮）
                
                # 启动后台线程处理
                thread = threading.Thread(
                    target=self._save_chat_async,
                    args=(user_id, session_id, user_message, assistant_response, 
                          should_generate_summary, final_state["step"] == "use_cache")
                )
                thread.daemon = True
                thread.start()
            
            return result
            
        except Exception as e:
            print(f"❌ 对话失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "response": "抱歉，我遇到了一些问题。让我们换个话题聊聊吧？"
            }
    
    def _save_chat_async(self, user_id: int, session_id: str, user_message: str, 
                         assistant_response: str, should_generate_summary: bool, used_cache: bool):
        """异步保存对话和生成总结"""
        try:
            context_summary = None
            
            # 只有在需要时才生成总结
            if should_generate_summary and self.db and LANGGRAPH_AVAILABLE and self.llm:
                try:
                    context_summary = self._generate_conversation_summary(
                        user_id, session_id, user_message, assistant_response
                    )
                except Exception as e:
                    print(f"⚠️ 生成总结失败（继续保存）: {e}")
            
            # 保存对话到数据库
            if self.db:
                self.db.save_conversation_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="user",
                    content=user_message,
                    context_summary=context_summary
                )
                self.db.save_conversation_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    content=assistant_response,
                    context_summary=context_summary
                )
                
                # 更新会话的最后消息时间
                self.db.update_session_last_message(session_id)
                
                # 保存会话总结（如果有）
                if context_summary:
                    self.db.save_session_summary(session_id, user_id, context_summary)
            
            print(f"✅ 对话异步保存完成")
            
        except Exception as e:
            print(f"❌ 异步保存失败: {e}")
            import traceback
            traceback.print_exc()


# 全局 Agent 实例
_langgraph_agent_instance: Optional[LangGraphEmotionAgent] = None


def get_langgraph_agent() -> LangGraphEmotionAgent:
    """获取或创建全局 LangGraph Agent 实例"""
    global _langgraph_agent_instance
    if _langgraph_agent_instance is None:
        _langgraph_agent_instance = LangGraphEmotionAgent()
    return _langgraph_agent_instance
