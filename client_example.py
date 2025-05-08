#!/usr/bin/env python
"""
宠物食谱优化API客户端

专门针对宠物食谱优化API设计的客户端，处理API响应格式和错误处理。
支持同步和异步优化请求，可视化结果，以及错误处理。

作者：
日期：2025-05-01
版本：1.1.0
"""

import os
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pet_recipe_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pet_recipe_client")

class APIError(Exception):
    """API错误类，包含状态码和详细信息"""
    def __init__(self, status_code: int, message: str, response_data: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.response_data = response_data
        super().__init__(f"API错误 ({status_code}): {message}")

class PetRecipeClient:
    """宠物食谱优化API客户端类"""
    
    def __init__(self, api_url: str, api_key: str, timeout: int = 30, max_retries: int = 3):
        """
        初始化API客户端
        
        参数:
        api_url: str - API基础URL，例如 'http://localhost:8000'
        api_key: str - API密钥
        timeout: int - 请求超时时间(秒)
        max_retries: int - 最大重试次数
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # 设置基本请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-API-Key': self.api_key
        })
        
        # 验证设置
        self._validate_settings()
    
    def _validate_settings(self) -> bool:
        """验证客户端设置"""
        if not self.api_url:
            raise ValueError("API URL不能为空")
        if not self.api_key:
            raise ValueError("API密钥不能为空")
        
        # 测试API连接
        try:
            health = self.check_health()
            logger.info(f"成功连接到API服务: {self.api_url}, 版本: {health.get('version', '未知')}")
            return True
        except Exception as e:
            logger.warning(f"API连接测试失败: {str(e)}")
            return False
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                params: Optional[Dict] = None, retry_count: int = 0) -> Dict[str, Any]:
        """
        发送API请求并处理响应
        
        参数:
        method: str - HTTP方法 ('GET', 'POST', etc.)
        endpoint: str - API端点，以'/'开头
        data: Dict - 请求体数据
        params: Dict - URL参数
        retry_count: int - 当前重试次数
        
        返回:
        Dict[str, Any] - API响应
        
        抛出:
        APIError - API错误
        ConnectionError - 连接错误
        TimeoutError - 请求超时
        """
        # 确保端点以'/'开头
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
            
        url = f"{self.api_url}{endpoint}"
        
        # 记录请求信息
        logger.debug(f"发送 {method} 请求: {url}")
        
        try:
            start_time = time.time()
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            
            logger.debug(f"API响应时间: {elapsed:.2f}秒, 状态码: {response.status_code}")
            
            # 尝试解析响应内容
            try:
                response_data = response.json()
            except ValueError:
                response_data = {"detail": "无法解析响应内容", "raw": response.text[:200]}
            
            # 检查响应状态
            if response.status_code >= 400:
                error_msg = response_data.get('detail', '未知错误')
                logger.error(f"API错误 ({response.status_code}): {error_msg}")
                
                # 处理特定错误类型
                if response.status_code == 401 or response.status_code == 403:
                    raise APIError(response.status_code, "认证失败，请检查API密钥", response_data)
                elif response.status_code == 404:
                    raise APIError(response.status_code, f"请求的资源不存在: {endpoint}", response_data)
                elif response.status_code == 422:
                    # 输入验证错误，提供更详细的错误信息
                    validation_errors = response_data.get("detail", [])
                    error_details = []
                    if isinstance(validation_errors, list):
                        for err in validation_errors[:3]:  # 只显示前3个错误
                            if isinstance(err, dict):
                                loc = ".".join(str(x) for x in err.get("loc", []))
                                msg = err.get("msg", "")
                                error_details.append(f"{loc}: {msg}")
                    
                    detailed_msg = "请求参数验证失败"
                    if error_details:
                        detailed_msg += f" - {', '.join(error_details)}"
                    
                    raise APIError(response.status_code, detailed_msg, response_data)
                elif response.status_code == 429:
                    # 限流错误，尝试重试
                    if retry_count < self.max_retries:
                        retry_after = int(response.headers.get('Retry-After', '5'))
                        logger.warning(f"API请求频率限制，等待 {retry_after} 秒后重试...")
                        time.sleep(retry_after)
                        return self._request(method, endpoint, data, params, retry_count + 1)
                    raise APIError(response.status_code, "超过API请求频率限制", response_data)
                elif response.status_code >= 500:
                    # 服务器错误，尝试重试
                    if retry_count < self.max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"服务器错误，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        return self._request(method, endpoint, data, params, retry_count + 1)
                    raise APIError(response.status_code, "服务器内部错误", response_data)
                else:
                    raise APIError(response.status_code, error_msg, response_data)
            
            return response_data
            
        except requests.Timeout:
            logger.error(f"请求超时: {url}")
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(f"请求超时，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                return self._request(method, endpoint, data, params, retry_count + 1)
            raise TimeoutError(f"请求超时，已尝试 {self.max_retries} 次重试")
            
        except requests.ConnectionError:
            logger.error(f"连接错误: {url}")
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(f"连接错误，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                return self._request(method, endpoint, data, params, retry_count + 1)
            raise ConnectionError(f"无法连接到API服务 {self.api_url}")
            
        except Exception as e:
            logger.error(f"请求异常: {str(e)}")
            raise
    
    def check_health(self) -> Dict[str, Any]:
        """检查API健康状态"""
        return self._request("GET", "/api/health")
    
    def get_nutrients(self) -> Dict[str, List[Dict]]:
        """获取支持的营养素列表"""
        return self._request("GET", "/api/nutrients")
    
    def optimize_recipe(self, 
                       target_energy: float, 
                       min_variety: int = 4, 
                       max_variety: int = 8,
                       allergy_foods: Optional[List[str]] = None, 
                       stage: str = "adult",
                       alpha: Optional[float] = None, 
                       beta: Optional[float] = None,
                       variety_factor: Optional[float] = None, 
                       custom_nutrients: Optional[Dict[str, Dict]] = None,
                       async_mode: bool = False,
                       wait_for_result: bool = True, 
                       poll_interval: int = 2,
                       timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        优化宠物食谱
        
        参数:
        target_energy: float - 目标能量(kcal)
        min_variety: int - 最少食物种类
        max_variety: int - 最多食物种类
        allergy_foods: List[str] - 过敏食物关键词列表
        stage: str - 宠物生命阶段，'adult'或'child'
        alpha: float - 能量偏差权重
        beta: float - 营养素松弛变量权重
        variety_factor: float - 食物多样性权重
        custom_nutrients: Dict - 自定义营养素需求
        async_mode: bool - 是否使用异步API
        wait_for_result: bool - 异步模式下是否等待结果
        poll_interval: int - 轮询间隔(秒)
        timeout: int - 等待超时时间(秒)
        
        返回:
        Dict[str, Any] - 优化结果
        """
        # 验证参数
        if target_energy <= 0:
            raise ValueError("目标能量必须为正数")
        if min_variety > max_variety:
            raise ValueError(f"最小食物种类({min_variety})不能大于最大食物种类({max_variety})")
        if stage not in ['adult', 'child']:
            raise ValueError("生命阶段必须是'adult'或'child'")
        
        # 准备请求数据
        request_data = {
            "target_energy": target_energy,
            "min_variety": min_variety,
            "max_variety": max_variety,
            "allergy_foods": allergy_foods or [],
            "stage": stage
        }
        
        # 添加可选参数
        if alpha is not None:
            request_data["alpha"] = alpha
        if beta is not None:
            request_data["beta"] = beta
        if variety_factor is not None:
            request_data["variety_factor"] = variety_factor
        if custom_nutrients is not None:
            request_data["custom_nutrients"] = custom_nutrients
        
        # 选择API端点
        endpoint = "/api/recipe/optimize-async" if async_mode else "/api/recipe/optimize"
        
        # 发送请求
        logger.info(f"发送{'异步' if async_mode else '同步'}优化请求: 目标能量={target_energy}kcal")
        response = self._request("POST", endpoint, data=request_data)
        
        # 同步API直接返回结果
        if not async_mode:
            return self._normalize_recipe_result(response)
        
        # 异步API返回任务ID
        task_id = response.get("task_id")
        if not task_id:
            raise ValueError("响应中缺少任务ID")
        
        logger.info(f"异步请求已提交，任务ID: {task_id}")
        
        # 如果不等待结果，直接返回任务信息
        if not wait_for_result:
            return response
        
        # 等待并获取结果
        result = self._poll_result(task_id, poll_interval, timeout)
        return self._normalize_recipe_result(result)
    
    def _poll_result(self, task_id: str, poll_interval: int = 2, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        轮询任务结果
        
        参数:
        task_id: str - 任务ID
        poll_interval: int - 轮询间隔(秒)
        timeout: int - 等待超时时间(秒)
        
        返回:
        Dict[str, Any] - 任务结果
        """
        start_time = time.time()
        
        logger.info(f"开始轮询任务结果: {task_id}")
        
        while True:
            # 检查是否超时
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"等待优化结果超时 ({timeout}秒)")
            
            try:
                # 获取结果
                result = self._request("GET", f"/api/recipe/result/{task_id}")
                
                # 检查任务是否完成
                if result.get("status") != "processing":
                    logger.info(f"任务完成: {task_id}, 状态: {result.get('status')}")
                    return result
                
                logger.debug(f"任务 {task_id} 正在处理中，等待 {poll_interval} 秒...")
                
            except APIError as e:
                if e.status_code == 404:
                    # 任务可能尚未创建完成，继续等待
                    logger.debug(f"任务 {task_id} 尚未准备好，等待 {poll_interval} 秒...")
                else:
                    # 其他API错误，直接抛出
                    raise
            
            # 等待一段时间再次轮询
            time.sleep(poll_interval)
    
    def _normalize_recipe_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化食谱结果，处理不同的数据格式
        
        参数:
        result: Dict[str, Any] - 原始结果
        
        返回:
        Dict[str, Any] - 标准化结果
        """
        if not result:
            return result
            
        normalized = result.copy()
        
        # 标准化selected_foods，统一为字典格式
        if 'selected_foods' in normalized:
            foods = []
            for food in normalized['selected_foods']:
                if isinstance(food, dict):
                    # 已经是字典格式
                    foods.append(food)
                elif isinstance(food, (list, tuple)) and len(food) >= 3:
                    # 转换元组/列表为字典
                    foods.append({
                        'id': str(food[0]),
                        'description': food[1],
                        'amount': food[2]
                    })
            normalized['selected_foods'] = foods
        
        # 标准化categories.*.foods，统一为字典格式
        if 'categories' in normalized:
            for category, info in normalized['categories'].items():
                if 'foods' in info:
                    cat_foods = []
                    for food in info['foods']:
                        if isinstance(food, dict):
                            # 已经是字典格式
                            cat_foods.append(food)
                        elif isinstance(food, (list, tuple)) and len(food) >= 3:
                            # 转换元组/列表为字典
                            cat_foods.append({
                                'id': str(food[0]),
                                'description': food[1],
                                'amount': food[2]
                            })
                    info['foods'] = cat_foods
        
        return normalized
    
    def get_result(self, result_id: str) -> Dict[str, Any]:
        """
        获取优化结果
        
        参数:
        result_id: str - 结果或任务ID
        
        返回:
        Dict[str, Any] - 优化结果
        """
        logger.info(f"获取结果: {result_id}")
        result = self._request("GET", f"/api/recipe/result/{result_id}")
        return self._normalize_recipe_result(result)
    
    def save_recipe(self, recipe: Dict[str, Any], file_path: str, format_type: str = "json") -> None:
        """
        保存食谱结果到文件
        
        参数:
        recipe: Dict[str, Any] - 食谱结果
        file_path: str - 保存路径
        format_type: str - 文件格式，'json'或'txt'
        """
        if not recipe:
            raise ValueError("食谱数据为空")
        
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format_type.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(recipe, f, ensure_ascii=False, indent=2)
            elif format_type.lower() == "txt":
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 写入基本信息
                    f.write(f"===== 宠物食谱优化结果 =====\n")
                    f.write(f"ID: {recipe.get('id', 'N/A')}\n")
                    f.write(f"状态: {recipe.get('status', 'N/A')}\n")
                    f.write(f"时间: {recipe.get('timestamp', 'N/A')}\n\n")
                    
                    # 写入能量信息
                    energy = recipe.get('energy', {})
                    f.write(f"能量信息:\n")
                    f.write(f"  目标能量: {energy.get('target', 'N/A')} kcal\n")
                    f.write(f"  实际能量: {energy.get('actual', 'N/A')} kcal\n")
                    f.write(f"  偏差: {energy.get('deviation', 'N/A')} kcal ({energy.get('percentage', 'N/A')}%)\n\n")
                    
                    # 写入食物信息
                    foods = recipe.get('selected_foods', [])
                    f.write(f"选定食物 ({len(foods)}种):\n")
                    for food in foods:
                        if isinstance(food, dict):
                            f.write(f"  - {food.get('description', 'N/A')}: {food.get('amount', 'N/A')}g\n")
                        elif isinstance(food, (list, tuple)) and len(food) >= 3:
                            f.write(f"  - {food[1]}: {food[2]}g\n")
                        else:
                            f.write(f"  - {food}\n")
                    
                    # 写入未满足的营养素
                    unsatisfied = recipe.get('unsatisfied_nutrients', [])
                    if unsatisfied:
                        f.write(f"\n未满足的营养素 ({len(unsatisfied)}个):\n")
                        for nut in unsatisfied:
                            f.write(f"  - {nut}\n")
            else:
                raise ValueError(f"不支持的文件格式: {format_type}")
            
            logger.info(f"食谱结果已保存到: {file_path}")
        
        except Exception as e:
            logger.error(f"保存食谱失败: {str(e)}")
            raise
    
    def visualize_recipe(self, recipe: Dict[str, Any], save_path: Optional[str] = None, show: bool = True) -> None:
        """
        可视化食谱结果
        
        参数:
        recipe: Dict[str, Any] - 食谱结果
        save_path: str - 图表保存路径，如果为None则不保存
        show: bool - 是否显示图表
        """
        if not recipe:
            raise ValueError("食谱数据为空")
        
        # 确保使用标准化的食谱数据
        recipe = self._normalize_recipe_result(recipe)
        
        try:
            # 设置中文字体(如果可用)
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            except:
                logger.warning("无法设置中文字体，图表中的中文可能无法正确显示")
            
            # 创建画布和子图布局
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 能量对比图
            self._plot_energy_comparison(axs[0, 0], recipe)
            
            # 2. 食物类别饼图
            self._plot_food_category_pie(axs[0, 1], recipe)
            
            # 3. 食物重量柱状图
            self._plot_food_weights(axs[1, 0], recipe)
            
            # 4. 营养素满足度条形图
            self._plot_nutrient_satisfaction(axs[1, 1], recipe)
            
            # 添加标题
            fig.suptitle(f"宠物食谱优化结果 - ID: {recipe.get('id', 'N/A')}", fontsize=16)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 保存图表
            if save_path:
                # 确保目录存在
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"图表已保存到: {save_path}")
            
            # 显示图表
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"可视化食谱失败: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)
            raise
    
    def _plot_energy_comparison(self, ax, recipe: Dict[str, Any]) -> None:
        """绘制能量对比图"""
        energy = recipe.get('energy', {})
        if not energy:
            ax.text(0.5, 0.5, '无能量数据', ha='center', va='center')
            ax.set_title('能量对比')
            return
            
        # 准备数据
        labels = ['目标能量', '实际能量']
        values = [energy.get('target', 0), energy.get('actual', 0)]
        colors = ['#3498db', '#2ecc71']
        
        # 绘制条形图
        bars = ax.bar(labels, values, color=colors)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                height * 1.01, 
                f'{height:.1f} kcal', 
                ha='center', 
                va='bottom'
            )
        
        # 设置标题和轴标签
        ax.set_title('能量对比')
        ax.set_ylabel('能量 (kcal)')
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加偏差说明
        deviation = energy.get('deviation', 0)
        percentage = energy.get('percentage', 0)
        deviation_text = f'偏差: {deviation:.2f} kcal ({percentage:.2f}%)'
        ax.annotate(deviation_text, xy=(0.5, 0.05), xycoords='axes fraction', 
                   ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                     fc='#f8f9fa', ec='#e9ecef'))
    
    def _plot_food_category_pie(self, ax, recipe: Dict[str, Any]) -> None:
        """绘制食物类别饼图"""
        categories = recipe.get('categories', {})
        if not categories:
            ax.text(0.5, 0.5, '无食物类别数据', ha='center', va='center')
            ax.set_title('食物类别分布')
            return
            
        # 准备数据
        labels = []
        sizes = []
        
        for cat, info in categories.items():
            cat_name = cat.capitalize()
            percentage = info.get('percentage', 0)
            food_count = len(info.get('foods', []))
            label = f"{cat_name} ({food_count}种, {percentage:.1f}%)"
            
            labels.append(label)
            sizes.append(percentage)
        
        # 绘制饼图
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
               wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        
        # 设置标题
        ax.set_title('食物类别分布')
        ax.axis('equal')  # 保持圆形
    
    def _plot_food_weights(self, ax, recipe: Dict[str, Any]) -> None:
        """绘制食物重量柱状图"""
        foods = recipe.get('selected_foods', [])
        if not foods:
            ax.text(0.5, 0.5, '无食物数据', ha='center', va='center')
            ax.set_title('食物重量分布')
            return
            
        # 准备数据
        descriptions = []
        weights = []
        
        for food in foods:
            if isinstance(food, dict):
                desc = food.get('description', '未知')
                amount = food.get('amount', 0)
            elif isinstance(food, (list, tuple)) and len(food) >= 3:
                desc = food[1]
                amount = food[2]
            else:
                continue
                
            # 截取描述，避免过长
            if len(desc) > 30:
                desc = desc[:27] + '...'
                
            descriptions.append(desc)
            weights.append(amount)
        
        # 按重量排序
        sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i])
        descriptions = [descriptions[i] for i in sorted_indices]
        weights = [weights[i] for i in sorted_indices]
        
        # 绘制水平条形图
        bars = ax.barh(descriptions, weights, color='#2ecc71')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}g', 
                va='center'
            )
        
        # 设置标题和轴标签
        ax.set_title('食物重量分布')
        ax.set_xlabel('重量 (克)')
        
        # 添加网格线
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    def _plot_nutrient_satisfaction(self, ax, recipe: Dict[str, Any]) -> None:
        """绘制营养素满足度条形图"""
        nutrients = recipe.get('nutrients', {})
        if not nutrients:
            ax.text(0.5, 0.5, '无营养素数据', ha='center', va='center')
            ax.set_title('营养素满足度')
            return
            
        # 准备数据
        names = []
        satisfactions = []
        
        # 选择前10个营养素，避免图表过于拥挤
        count = 0
        for _, info in sorted(nutrients.items(), key=lambda x: float(x[1].get('satisfaction', 0))):
            if count >= 10:
                break
                
            name = info.get('name', '未知')
            satisfaction = float(info.get('satisfaction', 0))
            
            # 截取名称，避免过长
            if len(name) > 20:
                name = name[:17] + '...'
                
            names.append(name)
            satisfactions.append(satisfaction)
            count += 1
        
        # 绘制水平条形图
        bars = ax.barh(names, satisfactions, color='#3498db')
        
        # 添加100%满足度垂直线
        ax.axvline(x=100, color='r', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                va='center'
            )
        
        # 设置标题和轴标签
        ax.set_title('营养素满足度')
        ax.set_xlabel('满足度百分比')
        
        # 设置x轴范围
        max_sat = max(satisfactions) if satisfactions else 100
        ax.set_xlim(0, max(120, max_sat * 1.1))
        
        # 添加网格线
        ax.grid(axis='x', linestyle='--', alpha=0.7)


def main():
    """
    命令行执行入口
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="宠物食谱优化API客户端")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API服务URL")
    parser.add_argument("--key", type=str, required=True, help="API密钥")
    parser.add_argument("--energy", type=float, required=True, help="目标能量(kcal)")
    parser.add_argument("--min-variety", type=int, default=4, help="最少食物种类")
    parser.add_argument("--max-variety", type=int, default=8, help="最多食物种类")
    parser.add_argument("--allergy", type=str, nargs="+", default=[], help="过敏食物列表")
    parser.add_argument("--stage", type=str, choices=["adult", "child"], default="adult", help="宠物生命阶段")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="使用异步API")
    parser.add_argument("--timeout", type=int, default=60, help="等待结果超时时间(秒)")
    parser.add_argument("--output", type=str, default="recipe_result.json", help="输出文件路径")
    parser.add_argument("--format", type=str, choices=["json", "txt"], default="json", help="输出文件格式")
    parser.add_argument("--visualize", action="store_true", help="可视化结果")
    parser.add_argument("--chart", type=str, default=None, help="图表保存路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 创建客户端
        client = PetRecipeClient(
            api_url=args.url,
            api_key=args.key,
            timeout=30
        )
        
        # 获取支持的营养素列表
        try:
            nutrients = client.get_nutrients()
            logger.info(f"API支持 {len(nutrients.get('nutrients', []))} 种营养素")
        except Exception as e:
            logger.warning(f"获取营养素列表失败: {str(e)}")
        
        # 优化食谱
        recipe = client.optimize_recipe(
            target_energy=args.energy,
            min_variety=args.min_variety,
            max_variety=args.max_variety,
            allergy_foods=args.allergy,
            stage=args.stage,
            async_mode=args.async_mode,
            timeout=args.timeout
        )
        
        # 打印结果摘要
        print("\n===== 宠物食谱优化结果 =====")
        print(f"ID: {recipe.get('id', 'N/A')}")
        print(f"状态: {recipe.get('status', 'N/A')}")
        print(f"执行时间: {recipe.get('execution_time', 0):.2f}秒")
        
        # 打印能量信息
        energy = recipe.get('energy', {})
        print(f"\n能量信息:")
        print(f"  目标能量: {energy.get('target', 'N/A')} kcal")
        print(f"  实际能量: {energy.get('actual', 'N/A')} kcal")
        print(f"  偏差: {energy.get('deviation', 'N/A')} kcal ({energy.get('percentage', 'N/A')}%)")
        
        # 打印选中的食物
        foods = recipe.get('selected_foods', [])
        print(f"\n选中食物 ({len(foods)}种):")
        for food in foods:
            if isinstance(food, dict):
                print(f"  - {food.get('description', 'N/A')}: {food.get('amount', 'N/A')}g")
            else:
                print(f"  - {food}")
        
        # 检查未满足的营养素
        unsatisfied = recipe.get('unsatisfied_nutrients', [])
        if unsatisfied:
            print(f"\n未满足的营养素 ({len(unsatisfied)}个):")
            for nut in unsatisfied:
                print(f"  - {nut}")
        else:
            print("\n所有营养素需求已满足")
        
        # 保存结果
        client.save_recipe(recipe, args.output, args.format)
        print(f"\n结果已保存到: {args.output}")
        
        # 可视化结果
        if args.visualize:
            chart_path = args.chart or os.path.splitext(args.output)[0] + ".png"
            client.visualize_recipe(recipe, chart_path)
            print(f"图表已保存到: {chart_path}")
        
        return 0
        
    except APIError as e:
        print(f"API错误 ({e.status_code}): {e.message}")
        if args.debug and e.response_data:
            print(f"详细信息: {json.dumps(e.response_data, ensure_ascii=False, indent=2)}")
        return 1
        
    except (ConnectionError, TimeoutError) as e:
        print(f"连接错误: {str(e)}")
        return 1
        
    except Exception as e:
        print(f"错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def simple_example():
    """
    简单使用示例
    """
    # 设置API参数
    api_url = "http://localhost:8000"  # 替换为实际API地址
    api_key = "your-secret-key-1"      # 替换为你的API密钥
    
    try:
        # 创建客户端
        client = PetRecipeClient(api_url, api_key)
        
        # 优化食谱
        recipe = client.optimize_recipe(
            target_energy=1500,         # 目标能量(kcal)
            min_variety=4,              # 最少食物种类
            max_variety=8,              # 最多食物种类
            allergy_foods=["pork"],     # 排除猪肉
            stage="adult",              # 成年犬
            async_mode=True             # 使用异步API避免因同步API的500错误
        )
        
        # 打印选中的食物
        print("\n选中的食物:")
        for food in recipe.get('selected_foods', []):
            print(f"- {food.get('description')}: {food.get('amount')}g")
        
        # 保存结果
        client.save_recipe(recipe, "recipe_result.json")
        client.save_recipe(recipe, "recipe_result.txt", "txt")
        
        # 可视化结果
        client.visualize_recipe(recipe, "recipe_chart.png")
        
        return recipe
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return None


if __name__ == "__main__":
    import sys
    sys.exit(main())