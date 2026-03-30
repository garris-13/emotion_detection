"""
数据库初始化脚本
用于创建数据库表和迁移现有数据
"""
import os
import sys
import json
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
import pymysql

# 加载环境变量
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def get_db_connection(database=None):
    """获取数据库连接"""
    host = os.getenv('MYSQL_HOST', 'localhost')
    port = int(os.getenv('MYSQL_PORT', '3306'))
    user = os.getenv('MYSQL_USER', 'root')
    password = os.getenv('MYSQL_PASSWORD', '')
    
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        raise


def create_database():
    """创建数据库"""
    print("=" * 70)
    print("🗄️ 创建数据库")
    print("=" * 70)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS emocare DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print("✅ 数据库创建成功")
    finally:
        conn.close()


def execute_schema():
    """执行数据库表结构"""
    print("\n" + "=" * 70)
    print("📋 创建数据库表")
    print("=" * 70)
    
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    conn = get_db_connection('emocare')
    try:
        with conn.cursor() as cursor:
            # 分割并执行 SQL 语句
            statements = []
            current_statement = []
            
            for line in schema_sql.split('\n'):
                line = line.strip()
                if line and not line.startswith('--'):
                    current_statement.append(line)
                    if line.endswith(';'):
                        statements.append(' '.join(current_statement))
                        current_statement = []
            
            for statement in statements:
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        print(f"⚠️ 执行语句时可能出现问题: {e}")
        
        conn.commit()
        print("✅ 数据库表创建成功")
        
        # 显示表
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"\n📊 已创建的表:")
            for table in tables:
                table_name = list(table.values())[0]
                print(f"  - {table_name}")
                
    finally:
        conn.close()


def migrate_existing_data():
    """迁移现有的 JSON 数据到数据库"""
    print("\n" + "=" * 70)
    print("🔄 迁移现有数据")
    print("=" * 70)
    
    from backend.database import get_db_manager
    
    db = get_db_manager()
    
    # 获取或创建默认用户
    user = db.get_or_create_user('default_user', 'user@example.com')
    user_id = user['id']
    print(f"✅ 用户ID: {user_id}")
    
    # 迁移监测数据
    results_dir = os.path.join(PROJECT_ROOT, 'data', 'monitor_results', 'results')
    
    if os.path.exists(results_dir):
        migrated_count = 0
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 保存到数据库
                    db.save_emotion_record(
                        user_id=user_id,
                        emotion_data=data,
                        session_id='migration'
                    )
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 迁移文件失败 {filename}: {e}")
        
        print(f"✅ 成功迁移 {migrated_count} 条监测数据")
    else:
        print("ℹ️ 未找到现有监测数据目录")
    
    print("\n" + "=" * 70)
    print("✅ 数据迁移完成")
    print("=" * 70)


def main():
    """主函数"""
    try:
        create_database()
        execute_schema()
        migrate_existing_data()
        print("\n🎉 数据库初始化完成！")
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
