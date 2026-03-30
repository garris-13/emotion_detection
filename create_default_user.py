
"""
创建默认用户脚本
"""
import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.database import get_db_manager

def create_default_user():
    """创建默认用户"""
    try:
        db_manager = get_db_manager()
        
        # 检查用户是否已存在
        existing_user = db_manager.get_user_by_username("default_user")
        if existing_user:
            print("用户 'default_user' 已存在")
            return existing_user
        
        # 创建默认用户
        user = db_manager.register_user(
            username="default_user",
            password="default123",
            email=None,
            age_group="adult",
            stress_level="medium",
            has_support_system=True
        )
        
        if user:
            print(f"✅ 默认用户创建成功: {user['username']}")
            print(f"   用户名: default_user")
            print(f"   密码: default123")
            return user
        else:
            print("❌ 创建默认用户失败")
            return None
        
    except Exception as e:
        print(f"❌ 创建默认用户失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("👤 创建默认用户")
    print("=" * 70)
    create_default_user()