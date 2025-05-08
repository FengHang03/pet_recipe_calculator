import os
from client_example import PetRecipeClient
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_client_connection():
    """
    测试客户端连接
    """
    try:
        # 设置环境变量
        os.environ['API_KEY'] = 'test_key'
        os.environ['SERVER_URL'] = 'http://localhost:8000'
        
        # 创建客户端实例
        client = PetRecipeClient()
        
        # 首先测试健康检查
        logger.info("Testing server health...")
        health_status = client.check_health()
        logger.info(f"Health check result: {health_status}")
        
        # 测试食谱优化
        logger.info("Testing recipe optimization...")
        result = client.optimize_recipe(
            target_energy=1000,
            min_variety=4,
            max_variety=8,
            allergy_foods=["chocolate", "coffee"],
            pet_type="adult"
        )
        
        # 打印结果
        logger.info("Connection successful!")
        logger.info("Optimization result:")
        logger.info(result)
        
        return True
        
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_client_connection() 