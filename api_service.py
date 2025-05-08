"""
宠物食谱优化API服务

本模块提供了基于FastAPI的API服务，用于将宠物食谱优化算法暴露为Web API。
包含了认证、限流、安全配置等功能，确保API安全可靠。

作者：
日期：2025-04-28
版本：1.0.0
"""

import time
import os
import secrets
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from functools import lru_cache
from datetime import datetime, timedelta

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS
from starlette.middleware.base import BaseHTTPMiddleware

from functools import lru_cache
import logging
from contextlib import asynccontextmanager

# 导入宠物食谱优化算法
from recipe_optimize import recipe_optimize, load_aafco_constraints, visualize_solution
from data_operate import get_all_allergic_fdc_ids

# 加载环境变量
load_dotenv()

# API密钥设置
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 从环境变量加载API密钥
API_KEYS = os.getenv("API_KEYS", "default-key").split(",")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pet_recipe_api")

# 结果缓存(简单内存缓存，生产环境可用Redis等)
result_cache = {}

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    logger.info("API service starting...")
    yield
    # 关闭事件
    logger.info("API service shutting down")
    result_cache.clear()

# 创建FastAPI应用
app = FastAPI(
    title="Pet Recipe Optimization API",
    description="API for optimizing pet food recipes based on nutritional requirements",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",  # Swagger UI路径
    redoc_url="/api/redoc",  # ReDoc路径
    openapi_url="/api/openapi.json"  # OpenAPI架构
)

# 添加CORS中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)  # 启用Gzip压缩
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8080").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 请求模型
class NutrientRequirement(BaseModel):
    """营养素需求模型"""
    min: Optional[float] = None
    max: Optional[float] = None
    unit: Optional[str] = None
    
    @validator('min', 'max')
    def validate_values(cls, v):
        if v is not None and v < 0:
            raise ValueError("营养素需求值不能为负数")
        return v
    
class RecipeRequest(BaseModel):
    """食谱优化请求模型"""
    target_energy: float = Field(..., description="目标能量(kcal)")
    min_variety: int = Field(4, description="最少食物种类")
    max_variety: int = Field(8, description="最多食物种类")
    allergy_foods: List[str] = Field([], description="过敏食物关键词列表")
    stage: str = Field("adult", description="宠物生命阶段，adult或child")
    alpha: Optional[float] = Field(None, description="能量偏差权重")
    beta: Optional[float] = Field(None, description="营养素松弛变量权重")
    variety_factor: Optional[float] = Field(None, description="食物多样性权重")
    custom_nutrients: Optional[Dict[str, NutrientRequirement]] = Field(None, description="自定义营养素需求")
    
    @validator('target_energy')
    def validate_energy(cls, v):
        if v <= 0:
            raise ValueError("目标能量必须为正数")
        return v
    
    @validator('min_variety', 'max_variety')
    def validate_variety(cls, v, values):
        if 'min_variety' in values and 'max_variety' in values:
            min_variety = values.get('min_variety', 4)
            max_variety = values.get('max_variety', 8)
            if min_variety > max_variety:
                raise ValueError(f"min_variety ({min_variety}) 不能大于 max_variety ({max_variety})")
        return v
    
    @validator('stage')
    def validate_stage(cls, v):
        if v not in ['adult', 'child']:
            raise ValueError("stage 必须是 'adult' 或 'child'")
        return v

class SelectedFood(BaseModel):
    """选中的食物模型"""
    id: str
    description: str
    amount: float
    category: Optional[str] = None

class NutrientInfo(BaseModel):
    """营养素信息模型"""
    id: int
    name: str
    actual: float
    required: float
    satisfaction: float
    slack: float
    unit: str

class EnergyInfo(BaseModel):
    """能量信息模型"""
    actual: float
    target: float
    deviation: float
    percentage: float

class CategoryInfo(BaseModel):
    """食物类别信息模型"""
    count: int
    weight: float
    percentage: float
    foods: List[Union[Dict[str, Any], Tuple[int, str, float]]]  # Accept both types

# 响应模型
class OptimizationResponse(BaseModel):
    success: bool
    message: str
    recipe: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: str

class RecipeResponse(BaseModel):
    """食谱优化响应模型"""
    id: str = Field(..., description="结果唯一ID")
    status: str = Field(..., description="优化状态") 
    timestamp: datetime = Field(..., description="处理时间")
    execution_time: float = Field(..., description="执行时间(秒)")
    selected_foods: List[Union[Dict[str, Any], Tuple[int, str, float]]] = Field(..., description="选中的食物列表")
    energy: EnergyInfo = Field(..., description="能量信息")
    nutrients: Dict[Union[str, int], NutrientInfo] = Field(..., description="营养素满足情况")
    unsatisfied_nutrients: List[str] = Field(..., description="未满足的营养素")
    categories: Dict[str, CategoryInfo] = Field(..., description="食物类别分布")
    message: Optional[str] = None

class AsyncTaskResponse(BaseModel):
    """异步任务响应模型"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="状态消息")
    result_url: Optional[str] = None

# 限流中间件
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 3600):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {api_key: [(timestamp), ...]}
    
    async def dispatch(self, request: Request, call_next):
        # 获取API密钥
        api_key = request.headers.get(API_KEY_NAME)
        
        # 如果没有API密钥，跳过限流检查
        if not api_key:
            return await call_next(request)
        
        now = time.time()
        
        # 清理过期的请求记录
        if api_key in self.requests:
            self.requests[api_key] = [r for r in self.requests[api_key] 
                                   if now - r < self.window_seconds]
        else:
            self.requests[api_key] = []
        
        # 检查请求频率
        if len(self.requests[api_key]) >= self.max_requests:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds} seconds."
                }
            )
        
        # 记录此次请求
        self.requests[api_key].append(now)
        
        # 继续处理请求
        return await call_next(request)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有API请求"""
    start_time = time.time()
    
    # 获取请求信息
    method = request.method
    url = request.url.path
    client_ip = request.client.host
    api_key = request.headers.get(API_KEY_NAME, "无API密钥")
    
    # 记录请求开始
    logger.info(f"开始处理请求: {method} {url} - 客户端: {client_ip}, API密钥: {api_key[:4]}***")
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 记录请求完成
    logger.info(f"请求处理完成: {method} {url} - 状态码: {response.status_code}, 处理时间: {process_time:.4f}秒")
    
    # 添加处理时间头
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 添加限流中间件
app.add_middleware(RateLimitMiddleware, max_requests=int(os.getenv("RATE_LIMIT", "100")))

# API密钥验证依赖
async def get_api_key(api_key_header: str = APIKeyHeader(name=API_KEY_NAME, auto_error=False)):
    """验证API密钥有效性"""
    if api_key_header in API_KEYS:
        return api_key_header
    
    # 记录无效的API密钥尝试
    logger.warning(f"API密钥验证失败: {api_key_header[:4]}***")
    
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, 
        detail="无效的API密钥"
    )

def sanitize_result(result: Dict) -> Dict:
    """
    处理结果，移除敏感信息，并格式化数据
    """
    sanitized = result.copy()
    
    # 删除模型对象（不能序列化且包含全部数据）
    if 'model' in sanitized:
        del sanitized['model']
    
    # 替换食物ID为哈希值，增加安全性
    if 'selected_foods' in sanitized:
        for i, food in enumerate(sanitized['selected_foods']):
            if isinstance(food, tuple) and len(food) >= 3:
                # 把元组转换为字典，并混淆ID
                food_id = str(food[0])
                hashed_id = hashlib.sha256(food_id.encode()).hexdigest()[:10]
                
                sanitized['selected_foods'][i] = {
                    'id': hashed_id,
                    'description': food[1],
                    'amount': food[2]
                }
    
    # 确保所有字段都是JSON可序列化的
    if 'timestamp' not in sanitized:
        sanitized['timestamp'] = datetime.now().isoformat()
    
    return sanitized

def convert_numpy_types(obj):
    """转换 NumPy 数据类型为 Python 原生类型"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

# 健康检查端点
@app.get("/")
async def root():
    """API状态检查"""
    return {
        "status": "online", 
        "service": "宠物食谱优化API", 
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# 食谱优化端点
@app.post("/api/recipe/optimize", response_model=RecipeResponse)
async def optimize_recipe(request: RecipeRequest, api_key: str = Depends(get_api_key)):
    """
    同步优化宠物食谱
    
    同步调用食谱优化算法，生成平衡的宠物食谱。如果请求可能需要长时间处理，
    建议使用异步API端点。
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 生成请求ID
        request_id = secrets.token_hex(8)
        logger.info(f"处理优化请求: {request_id} - 目标能量: {request.target_energy}kcal")
        
        # 获取过敏食物ID
        allergy_food_ids = list(get_all_allergic_fdc_ids(request.allergy_foods))
        logger.info(f"请求 {request_id}: 找到 {len(allergy_food_ids)} 个匹配过敏食物关键词的ID")
        
        # 获取营养需求
        nut_requirements = load_aafco_constraints(request.stage)
        
        # 合并自定义营养需求
        if request.custom_nutrients:
            for nut_id_str, req in request.custom_nutrients.items():
                try:
                    nut_id = int(nut_id_str)
                    nut_requirements[nut_id] = {"min": req.min, "max": req.max, "unit": req.unit}
                except ValueError:
                    logger.warning(f"请求 {request_id}: 忽略无效的营养素ID: {nut_id_str}")
        
        # 调用优化函数
        try:
            result = recipe_optimize(
                nut_requirements=nut_requirements,
                target_energy=request.target_energy,
                min_variety=request.min_variety,
                max_variety=request.max_variety,
                allergy_foods_id=allergy_food_ids,
                alpha=request.alpha,
                beta=request.beta,
                variety_factor=request.variety_factor,
                log_level="INFO"
            )
        except Exception as e:
            logger.exception(f"请求 {request_id}: 优化函数异常 - {str(e)}")
            result = {
                'status': 'Error',
                'error_message': f"优化过程中发生异常: {str(e)}",
                'selected_foods': [],
                'energy': {'actual': 0, 'target': request.target_energy, 'deviation': request.target_energy},
                'nutrients': {},
                'unsatisfied_nutrients': [],
                'categories': {}
            }
        
        # 处理结果
        if result['status'] == 'Error':
            logger.error(f"请求 {request_id}: 优化失败 - {result.get('error_message', '未知错误')}")
            raise HTTPException(status_code=500, detail=result.get('error_message', '优化过程中发生错误'))
        
        # 处理成功结果
        sanitized_result = sanitize_result(result)
        
        # 转换 NumPy 数据类型
        sanitized_result = convert_numpy_types(sanitized_result)
        
        # 添加结果元数据
        sanitized_result['id'] = request_id
        sanitized_result['timestamp'] = datetime.now()
        sanitized_result['execution_time'] = time.time() - start_time
        
        # 添加能量偏差百分比
        if 'energy' in sanitized_result:
            target = sanitized_result['energy']['target']
            deviation = sanitized_result['energy']['deviation']
            percentage = (deviation / target * 100) if target > 0 else 0
            sanitized_result['energy']['percentage'] = round(percentage, 2)
        
        # 缓存结果
        result_cache[request_id] = sanitized_result
        
        logger.info(f"请求 {request_id}: 优化成功，处理时间: {sanitized_result['execution_time']:.2f}秒")
        
        return sanitized_result
        
    except Exception as e:
        logger.exception(f"请求处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")
    
@app.post("/api/recipe/optimize-async", response_model=AsyncTaskResponse)
async def optimize_recipe_async(
    request: RecipeRequest, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    异步优化宠物食谱
    
    提交食谱优化请求，立即返回任务ID，后台处理优化任务。
    适用于复杂优化任务和批量处理。
    """
    # 生成任务ID
    task_id = secrets.token_hex(8)
    
    # 添加后台任务
    background_tasks.add_task(
        process_optimization_task,
        task_id=task_id,
        request=request,
        api_key=api_key
    )
    
    logger.info(f"提交异步任务: {task_id} - 目标能量: {request.target_energy}kcal")
    
    # 返回任务信息
    return AsyncTaskResponse(
        task_id=task_id,
        status="processing",
        message="优化任务已提交，正在后台处理",
        result_url=f"/api/recipe/result/{task_id}"
    )

@app.get("/api/recipe/result/{result_id}", response_model=RecipeResponse)
async def get_optimization_result(result_id: str, api_key: str = Depends(get_api_key)):
    """
    获取优化结果
    
    根据结果ID检索之前的优化结果
    """
    # 检查缓存
    if result_id in result_cache:
        return result_cache[result_id]
    
    # 结果不存在
    logger.warning(f"请求结果不存在: {result_id}")
    raise HTTPException(status_code=404, detail=f"找不到指定ID的结果: {result_id}")

@app.get("/api/nutrients")
async def get_nutrients(api_key: str = Depends(get_api_key)):
    """
    获取支持的营养素列表
    
    返回系统支持的所有营养素ID、名称和单位
    """
    from recipe_optimize import NutrientID
    
    # 获取成年犬的营养需求标准
    adult_requirements = load_aafco_constraints("adult")
    
    # 准备营养素列表
    nutrients = []
    
    # 获取NutrientID类的所有属性
    for attr_name in dir(NutrientID):
        # 跳过内置属性和方法
        if attr_name.startswith('__') or callable(getattr(NutrientID, attr_name)):
            continue
        
        # 获取营养素ID
        nutrient_id = getattr(NutrientID, attr_name)
        
        # 获取营养素名称（将常量名称转换为可读形式）
        name = " ".join(attr_name.lower().split('_'))
        
        # 获取单位和需求
        if nutrient_id in adult_requirements:
            unit = adult_requirements[nutrient_id].get('unit', '未知')
            min_val = adult_requirements[nutrient_id].get('min')
            max_val = adult_requirements[nutrient_id].get('max')
        else:
            unit = '未知'
            min_val = None
            max_val = None
        
        # 添加到列表
        nutrients.append({
            "id": nutrient_id,
            "name": name,
            "unit": unit,
            "min": min_val,
            "max": max_val
        })
    
    return {"nutrients": nutrients}

@app.get("/api/health")
async def health_check():
    """
    健康检查端点
    
    用于监控系统和负载均衡器检查API状态
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# 后台任务：处理优化任务
async def process_optimization_task(task_id: str, request: RecipeRequest, api_key: str):
    """后台处理优化任务"""
    try:
        # 记录开始时间
        start_time = time.time()
        logger.info(f"开始处理异步任务: {task_id}")
        
        # 获取过敏食物ID
        allergy_food_ids = list(get_all_allergic_fdc_ids(request.allergy_foods))
        
        # 获取营养需求
        nut_requirements = load_aafco_constraints(request.stage)
        
        # 合并自定义营养需求
        if request.custom_nutrients:
            for nut_id_str, req in request.custom_nutrients.items():
                try:
                    nut_id = int(nut_id_str)
                    nut_requirements[nut_id] = {"min": req.min, "max": req.max, "unit": req.unit}
                except ValueError:
                    logger.warning(f"任务 {task_id}: 忽略无效的营养素ID: {nut_id_str}")
        
        # 调用优化函数
        result = recipe_optimize(
            nut_requirements=nut_requirements,
            target_energy=request.target_energy,
            min_variety=request.min_variety,
            max_variety=request.max_variety,
            allergy_foods_id=allergy_food_ids,
            alpha=request.alpha,
            beta=request.beta,
            variety_factor=request.variety_factor,
            log_level="INFO"
        )
        
        # 处理和缓存结果
        sanitized_result = sanitize_result(result)
        sanitized_result['id'] = task_id
        sanitized_result['timestamp'] = datetime.now()
        sanitized_result['execution_time'] = time.time() - start_time
        
        # 转换 NumPy 数据类型
        sanitized_result = convert_numpy_types(sanitized_result)
        
        # 缓存结果
        result_cache[task_id] = sanitized_result
        
        logger.info(f"异步任务完成: {task_id}, 处理时间: {sanitized_result['execution_time']:.2f}秒")
        
    except Exception as e:
        logger.exception(f"异步任务处理异常: {task_id} - {str(e)}")
        
        # 缓存错误结果
        error_result = {
            'id': task_id,
            'status': 'Error',
            'message': f"处理失败: {str(e)}",
            'timestamp': datetime.now(),
            'execution_time': time.time() - start_time,
            'selected_foods': [],
            'energy': {'actual': 0, 'target': request.target_energy, 'deviation': request.target_energy, 'percentage': 100},
            'nutrients': {},
            'unsatisfied_nutrients': [],
            'categories': {}
        }
        
        result_cache[task_id] = error_result

# 启动任务：清理过期结果缓存
@app.on_event("startup")
async def setup_cache_cleanup():
    """设置定期清理过期缓存的任务"""
    import asyncio
    
    async def cleanup_expired_results():
        """清理过期的结果缓存"""
        while True:
            try:
                # 等待一段时间
                await asyncio.sleep(3600)  # 每小时执行一次
                
                # 获取当前时间
                now = datetime.now()
                expired_keys = []
                
                # 寻找过期的结果
                for key, result in result_cache.items():
                    if 'timestamp' in result:
                        timestamp = result['timestamp']
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        
                        # 24小时后过期
                        if now - timestamp > timedelta(hours=24):
                            expired_keys.append(key)
                
                # 删除过期结果
                for key in expired_keys:
                    del result_cache[key]
                
                if expired_keys:
                    logger.info(f"已清理 {len(expired_keys)} 个过期结果")
                    
            except Exception as e:
                logger.exception(f"缓存清理任务异常: {str(e)}")
    
    # 启动清理任务
    asyncio.create_task(cleanup_expired_results())

# 主函数：当直接运行此文件时执行
if __name__ == "__main__":
    # 从环境变量读取端口，默认为8000
    port = int(os.getenv("PORT", "8000"))
    
    # 启动服务器
    uvicorn.run(
        "api_service:app",  # 确保这里与你的文件名匹配
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        workers=int(os.getenv("WORKERS", "1"))
    )
