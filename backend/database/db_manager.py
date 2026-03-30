"""
数据库管理模块
用于操作 MySQL 数据库
"""
import os
import json
import pymysql
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# 尝试导入 bcrypt，如果失败则使用简单的密码处理
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    print("⚠️ bcrypt 不可用，将使用简单密码处理")

# 加载环境变量
load_dotenv()


class DatabaseManager:
    def __init__(self):
        self.host = os.getenv('MYSQL_HOST', 'localhost')
        self.port = int(os.getenv('MYSQL_PORT', '3306'))
        self.user = os.getenv('MYSQL_USER', 'root')
        self.password = os.getenv('MYSQL_PASSWORD', '')
        self.database = os.getenv('MYSQL_DATABASE', 'emocare')
        self.connection = None
        self._connect()

    def _connect(self):
        """连接数据库"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print(f"✅ 数据库连接成功: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            self.connection = None

    def _reconnect(self):
        """重新连接数据库"""
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
        self._connect()

    def _execute_query(self, query: str, params: tuple = None, fetch: bool = False, commit: bool = False):
        """执行查询"""
        if not self.connection:
            self._reconnect()
            if not self.connection:
                raise Exception("数据库连接不可用")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
                if fetch:
                    result = cursor.fetchall()
                else:
                    result = cursor.lastrowid
                if commit:
                    self.connection.commit()
                return result
        except pymysql.MySQLError as e:
            print(f"❌ 数据库查询失败: {e}")
            self.connection.rollback()
            self._reconnect()
            raise
        except Exception as e:
            print(f"❌ 查询执行失败: {e}")
            raise

    # ==================== 用户操作 ====================

    def get_or_create_user(self, username: str = 'default_user', email: str = None) -> Dict:
        """获取或创建用户"""
        query = "SELECT * FROM users WHERE username = %s"
        users = self._execute_query(query, (username,), fetch=True)

        if users:
            return users[0]

        insert_query = """
            INSERT INTO users (username, password, email, age_group, stress_level, has_support_system)
            VALUES (%s, %s, %s, 'adult', 'medium', TRUE)
        """
        hashed_password = self._hash_password('default123')
        user_id = self._execute_query(insert_query, (username, hashed_password, email), commit=True)

        return self.get_user_by_id(user_id)

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """根据ID获取用户"""
        query = "SELECT * FROM users WHERE id = %s"
        users = self._execute_query(query, (user_id,), fetch=True)
        return users[0] if users else None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """根据用户名获取用户"""
        query = "SELECT * FROM users WHERE username = %s"
        users = self._execute_query(query, (username,), fetch=True)
        return users[0] if users else None

    def _hash_password(self, password: str) -> str:
        """加密密码"""
        if BCRYPT_AVAILABLE:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        else:
            # 简单的密码哈希（仅用于开发测试）
            import hashlib
            return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        if BCRYPT_AVAILABLE:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        else:
            # 简单的密码验证（仅用于开发测试）
            import hashlib
            return hashlib.sha256(password.encode('utf-8')).hexdigest() == hashed_password

    def register_user(self, username: str, password: str, email: str = None,
                      age_group: str = 'adult', stress_level: str = 'medium',
                      has_support_system: bool = True) -> Optional[Dict]:
        """
        注册新用户
        
        Args:
            username: 用户名
            password: 密码（明文）
            email: 邮箱（可选）
            age_group: 年龄组
            stress_level: 压力水平
            has_support_system: 是否有社会支持系统
            
        Returns:
            成功返回用户信息，失败返回None
        """
        # 检查用户名是否已存在
        existing_user = self.get_user_by_username(username)
        if existing_user:
            print(f"❌ 用户名 '{username}' 已存在")
            return None

        # 加密密码
        hashed_password = self._hash_password(password)

        try:
            insert_query = """
                INSERT INTO users (username, password, email, age_group, stress_level, has_support_system)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            user_id = self._execute_query(
                insert_query,
                (username, hashed_password, email, age_group, stress_level, has_support_system),
                commit=True
            )
            print(f"✅ 用户 '{username}' 注册成功")
            return self.get_user_by_id(user_id)
        except Exception as e:
            print(f"❌ 用户注册失败: {e}")
            return None

    def login_user(self, username: str, password: str) -> Optional[Dict]:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码（明文）
            
        Returns:
            成功返回用户信息（不包含密码），失败返回None
        """
        user = self.get_user_by_username(username)
        if not user:
            print(f"❌ 用户 '{username}' 不存在")
            return None

        # 验证密码
        if self._verify_password(password, user.get('password', '')):
            # 返回用户信息，但不包含密码
            user_data = {k: v for k, v in user.items() if k != 'password'}
            print(f"✅ 用户 '{username}' 登录成功")
            return user_data
        else:
            print(f"❌ 用户 '{username}' 密码错误")
            return None

    # ==================== 情绪记录操作 ====================

    def save_emotion_record(self, user_id: int, emotion_data: Dict, session_id: str = None) -> int:
        """保存情绪记录"""
        # 首先检查用户是否存在，如果不存在则创建默认用户
        user = self.get_user_by_id(user_id)
        if not user:
            print(f"⚠️ 用户ID {user_id} 不存在，创建默认用户...")
            # 创建默认用户
            default_username = f"user_{user_id}"
            insert_query = """
                INSERT INTO users (id, username, password, age_group, stress_level, has_support_system)
                VALUES (%s, %s, %s, 'adult', 'medium', TRUE)
            """
            try:
                hashed_password = self._hash_password('default123')
                self._execute_query(insert_query, (user_id, default_username, hashed_password), commit=True)
                print(f"✅ 创建默认用户成功: {default_username} (ID: {user_id})")
            except Exception as e:
                print(f"⚠️ 创建默认用户失败: {e}，尝试使用自动生成的ID")
                # 如果指定ID插入失败，让数据库自动生成ID
                self.get_or_create_user(default_username)
        
        query = """
            INSERT INTO emotion_records 
            (user_id, emotion, emotion_zh, confidence, probabilities, image_path, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        record_id = self._execute_query(query, (
            user_id,
            emotion_data.get('emotion'),
            emotion_data.get('emotion_zh'),
            emotion_data.get('confidence'),
            json.dumps(emotion_data.get('probabilities', {})),
            emotion_data.get('image_path'),
            session_id
        ), commit=True)
        return record_id

    def get_emotion_records(self, user_id: int, days: int = None, 
                           start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """获取情绪记录"""
        query = "SELECT * FROM emotion_records WHERE user_id = %s"
        params = [user_id]

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            query += " AND recorded_at >= %s"
            params.append(cutoff)
        elif start_date and end_date:
            query += " AND recorded_at BETWEEN %s AND %s"
            params.extend([start_date, end_date])

        query += " ORDER BY recorded_at DESC"
        return self._execute_query(query, tuple(params), fetch=True)

    # ==================== 性格分析操作 ====================

    def save_personality_analysis(self, user_id: int, analysis_period: str,
                                   start_date: datetime, end_date: datetime,
                                   personality_summary: str, emotion_patterns: Dict,
                                   dominant_emotions: Dict, stability_score: float) -> int:
        """保存性格分析"""
        query = """
            INSERT INTO user_personality_analysis 
            (user_id, analysis_period, start_date, end_date, personality_summary, 
             emotion_patterns, dominant_emotions, stability_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        analysis_id = self._execute_query(query, (
            user_id,
            analysis_period,
            start_date.date(),
            end_date.date(),
            personality_summary,
            json.dumps(emotion_patterns),
            json.dumps(dominant_emotions),
            stability_score
        ), commit=True)
        return analysis_id

    def get_latest_personality_analysis(self, user_id: int, analysis_period: str = '6months') -> Optional[Dict]:
        """获取最新的性格分析"""
        query = """
            SELECT * FROM user_personality_analysis 
            WHERE user_id = %s AND analysis_period = %s
            ORDER BY generated_at DESC LIMIT 1
        """
        analyses = self._execute_query(query, (user_id, analysis_period), fetch=True)
        if analyses:
            analysis = analyses[0]
            if analysis.get('emotion_patterns'):
                analysis['emotion_patterns'] = json.loads(analysis['emotion_patterns'])
            if analysis.get('dominant_emotions'):
                analysis['dominant_emotions'] = json.loads(analysis['dominant_emotions'])
        return analyses[0] if analyses else None

    # ==================== 对话历史操作 ====================

    def save_conversation_message(self, user_id: int, session_id: str,
                                    role: str, content: str, context_summary: str = None) -> int:
        """保存对话消息"""
        query = """
            INSERT INTO conversation_history 
            (user_id, session_id, role, content, context_summary)
            VALUES (%s, %s, %s, %s, %s)
        """
        message_id = self._execute_query(query, (
            user_id, session_id, role, content, context_summary
        ), commit=True)
        return message_id

    def get_conversation_history(self, user_id: int, session_id: str, limit: int = 50) -> List[Dict]:
        """获取对话历史"""
        query = """
            SELECT * FROM conversation_history 
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at ASC LIMIT %s
        """
        return self._execute_query(query, (user_id, session_id, limit), fetch=True)

    # ==================== 对话会话操作 ====================

    def create_conversation_session(self, user_id: int, title: str) -> Optional[Dict]:
        """创建对话会话"""
        import uuid
        session_id = str(uuid.uuid4())
        query = """
            INSERT INTO conversation_sessions (id, user_id, title)
            VALUES (%s, %s, %s)
        """
        try:
            self._execute_query(query, (session_id, user_id, title), commit=True)
            return self.get_conversation_session(session_id)
        except Exception as e:
            print(f"❌ 创建会话失败: {e}")
            return None

    def get_conversation_session(self, session_id: str) -> Optional[Dict]:
        """获取单个会话"""
        query = "SELECT * FROM conversation_sessions WHERE id = %s"
        sessions = self._execute_query(query, (session_id,), fetch=True)
        return sessions[0] if sessions else None

    def get_user_conversation_sessions(self, user_id: int) -> List[Dict]:
        """获取用户的所有会话"""
        query = """
            SELECT * FROM conversation_sessions 
            WHERE user_id = %s 
            ORDER BY last_message_at DESC
        """
        return self._execute_query(query, (user_id,), fetch=True)

    def update_conversation_session(self, session_id: str, title: str = None) -> bool:
        """更新会话"""
        if title is None:
            return False
        query = """
            UPDATE conversation_sessions 
            SET title = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        try:
            self._execute_query(query, (title, session_id), commit=True)
            return True
        except Exception as e:
            print(f"❌ 更新会话失败: {e}")
            return False

    def update_session_last_message(self, session_id: str) -> bool:
        """更新会话的最后消息时间"""
        query = """
            UPDATE conversation_sessions 
            SET last_message_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        try:
            self._execute_query(query, (session_id,), commit=True)
            return True
        except Exception as e:
            print(f"❌ 更新会话时间失败: {e}")
            return False

    def delete_conversation_session(self, session_id: str) -> bool:
        """删除会话（级联删除所有相关数据）"""
        query = "DELETE FROM conversation_sessions WHERE id = %s"
        try:
            self._execute_query(query, (session_id,), commit=True)
            return True
        except Exception as e:
            print(f"❌ 删除会话失败: {e}")
            return False

    # ==================== 会话总结操作 ====================

    def save_session_summary(self, session_id: str, user_id: int, summary_text: str) -> Optional[Dict]:
        """保存会话总结"""
        # 获取当前最大的summary_order
        max_order_query = """
            SELECT MAX(summary_order) as max_order 
            FROM session_summaries 
            WHERE session_id = %s
        """
        result = self._execute_query(max_order_query, (session_id,), fetch=True)
        summary_order = (result[0]['max_order'] or 0) + 1

        insert_query = """
            INSERT INTO session_summaries (session_id, user_id, summary_order, summary_text)
            VALUES (%s, %s, %s, %s)
        """
        try:
            summary_id = self._execute_query(insert_query, (session_id, user_id, summary_order, summary_text), commit=True)
            # 获取刚插入的总结
            get_query = "SELECT * FROM session_summaries WHERE id = %s"
            summaries = self._execute_query(get_query, (summary_id,), fetch=True)
            return summaries[0] if summaries else None
        except Exception as e:
            print(f"❌ 保存会话总结失败: {e}")
            return None

    def get_session_summaries(self, session_id: str, limit: int = 2) -> List[Dict]:
        """获取会话的最近总结（默认最近2个）"""
        query = """
            SELECT * FROM session_summaries 
            WHERE session_id = %s 
            ORDER BY summary_order DESC 
            LIMIT %s
        """
        summaries = self._execute_query(query, (session_id, limit), fetch=True)
        # 按正序返回
        return list(reversed(summaries))

    def get_all_session_summaries(self, session_id: str) -> List[Dict]:
        """获取会话的所有总结"""
        query = """
            SELECT * FROM session_summaries 
            WHERE session_id = %s 
            ORDER BY summary_order ASC
        """
        return self._execute_query(query, (session_id,), fetch=True)

    # ==================== 系统提示词缓存操作 ====================

    def save_system_prompt_cache(self, user_id: int, session_id: str,
                                  personality_summary: str, recent_emotion_summary: str,
                                  full_system_prompt: str, expires_hours: int = 24) -> int:
        """保存系统提示词缓存"""
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        query = """
            INSERT INTO system_prompt_cache 
            (user_id, session_id, personality_summary, recent_emotion_summary, full_system_prompt, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cache_id = self._execute_query(query, (
            user_id, session_id, personality_summary, recent_emotion_summary, full_system_prompt, expires_at
        ), commit=True)
        return cache_id

    def get_system_prompt_cache(self, user_id: int, session_id: str) -> Optional[Dict]:
        """获取有效的系统提示词缓存"""
        query = """
            SELECT * FROM system_prompt_cache 
            WHERE user_id = %s AND session_id = %s AND expires_at > NOW()
            ORDER BY created_at DESC LIMIT 1
        """
        caches = self._execute_query(query, (user_id, session_id), fetch=True)
        return caches[0] if caches else None

    def clear_expired_cache(self):
        """清理过期的缓存"""
        query = "DELETE FROM system_prompt_cache WHERE expires_at <= NOW()"
        self._execute_query(query, commit=True)
        print("✅ 已清理过期缓存")

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("✅ 数据库连接已关闭")


# 全局数据库管理器实例
_db_instance: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取或创建数据库管理器实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
