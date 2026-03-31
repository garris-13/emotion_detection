-- ========================================================
-- EmoCare 数据库表结构
-- 用于存储用户情绪监测数据和对话记录
-- ========================================================

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS emocare DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE emocare;

-- ========================================================
-- 1. 用户表
-- ========================================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL COMMENT '用户名',
    password VARCHAR(255) NOT NULL COMMENT '密码（加密存储）',
    email VARCHAR(255) UNIQUE COMMENT '邮箱',
    age_group ENUM('child', 'teen', 'adult', 'elder') DEFAULT 'adult' COMMENT '年龄组',
    stress_level ENUM('low', 'medium', 'high') DEFAULT 'medium' COMMENT '压力水平',
    has_support_system BOOLEAN DEFAULT TRUE COMMENT '是否有社会支持系统',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- ========================================================
-- 2. 对话会话表
-- ========================================================
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id VARCHAR(100) PRIMARY KEY COMMENT '会话ID',
    user_id INT NOT NULL COMMENT '用户ID',
    title VARCHAR(255) NOT NULL COMMENT '会话标题',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '最后消息时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话会话表';

-- ========================================================
-- 3. 情绪监测记录表
-- ========================================================
CREATE TABLE IF NOT EXISTS emotion_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL COMMENT '用户ID',
    emotion VARCHAR(50) NOT NULL COMMENT '主要情绪英文',
    emotion_zh VARCHAR(50) NOT NULL COMMENT '主要情绪中文',
    confidence DECIMAL(5,4) NOT NULL COMMENT '置信度',
    probabilities JSON COMMENT '各情绪概率分布JSON',
    image_path VARCHAR(500) COMMENT '图片存储路径',
    session_id VARCHAR(100) COMMENT '会话ID',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_recorded_at (recorded_at),
    INDEX idx_emotion (emotion)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='情绪监测记录表';

-- ========================================================
-- 4. 用户性格分析表
-- ========================================================
CREATE TABLE IF NOT EXISTS user_personality_analysis (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL COMMENT '用户ID',
    analysis_period ENUM('1month', '6months', '1year') NOT NULL COMMENT '分析周期',
    start_date DATE NOT NULL COMMENT '分析开始日期',
    end_date DATE NOT NULL COMMENT '分析结束日期',
    personality_summary TEXT COMMENT '性格总结',
    emotion_patterns JSON COMMENT '情绪模式JSON',
    dominant_emotions JSON COMMENT '主导情绪JSON',
    stability_score DECIMAL(5,2) COMMENT '稳定性评分',
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '生成时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_generated_at (generated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户性格分析表';

-- ========================================================
-- 5. 对话历史表
-- ========================================================
CREATE TABLE IF NOT EXISTS conversation_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL COMMENT '用户ID',
    session_id VARCHAR(100) NOT NULL COMMENT '会话ID',
    role ENUM('user', 'assistant', 'system') NOT NULL COMMENT '角色',
    content TEXT NOT NULL COMMENT '内容',
    context_summary TEXT COMMENT '上下文总结',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话历史表';

-- ========================================================
-- 6. 会话总结表
-- ========================================================
CREATE TABLE IF NOT EXISTS session_summaries (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL COMMENT '会话ID',
    user_id INT NOT NULL COMMENT '用户ID',
    summary_order INT NOT NULL COMMENT '总结顺序（第几个总结）',
    summary_text TEXT NOT NULL COMMENT '总结内容',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_user_id (user_id),
    INDEX idx_summary_order (summary_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话总结表';

-- ========================================================
-- 7. 系统提示词缓存表
-- ========================================================
CREATE TABLE IF NOT EXISTS system_prompt_cache (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL COMMENT '用户ID',
    session_id VARCHAR(100) NOT NULL COMMENT '会话ID',
    personality_summary TEXT COMMENT '性格总结',
    recent_emotion_summary TEXT COMMENT '近期情绪总结',
    full_system_prompt TEXT COMMENT '完整系统提示词',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    expires_at TIMESTAMP COMMENT '过期时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_expires_at (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统提示词缓存表';

-- ========================================================
-- 为现有用户表添加密码字段（如果表已存在）
-- ========================================================
DELIMITER //

DROP PROCEDURE IF EXISTS add_password_column//
CREATE PROCEDURE add_password_column()
BEGIN
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION BEGIN END;
    ALTER TABLE users 
    ADD COLUMN password VARCHAR(255) NOT NULL COMMENT '密码（加密存储）' 
    AFTER username;
END//

DELIMITER ;

CALL add_password_column();
DROP PROCEDURE IF EXISTS add_password_column;

-- ========================================================
-- 显示创建结果
-- ========================================================
SHOW TABLES;

SELECT '数据库表创建完成！' AS message;
