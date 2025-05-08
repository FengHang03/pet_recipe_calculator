"""
宠物食谱优化模块 - 使用线性规划为宠物生成营养均衡的食谱

本模块实现了一个基于线性规划的宠物食谱优化算法，可以根据指定的营养需求、
目标能量以及各种约束条件（如食物类别比例、种类多样性等）生成最优的宠物食谱。

作者：
日期：2025-04-27
版本：2.0.0
"""

import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Union, Any
from calculate_ME import Pet
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, PULP_CBC_CMD, lpSum
import logging
import numpy as np
import os
import re
from functools import lru_cache
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
import configparser

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pet_recipe.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 从 data_operate.py 导入其他必要的函数
from data_operate import AAFCO_constraints

# 获取项目根目录，确保文件路径一致
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data'

# 确保数据目录存在
DATA_DIR.mkdir(exist_ok=True)

# 配置文件路径
CONFIG_PATH = PROJECT_ROOT / "config.ini"

@dataclass
class NutrientInfo:
    """营养素信息数据类"""
    id: int
    name: str
    amount: float
    unit: str

# 定义营养素常量
class NutrientID:
    """营养素ID常量定义"""
    ENERGY = 1008      # 能量 (kcal)
    PROTEIN = 1003     # 蛋白质 (g)
    CALCIUM = 1087     # 钙 (mg)
    PHOSPHORUS = 1091  # 磷 (mg)
    FAT = 1004         # 脂肪 (g)
    CARBOHYDRATE = 1005 # 碳水化合物 (g)
    IRON = 1089        # 铁 (mg)
    ZINC = 1095        # 锌 (mg)
    COPPER = 1098      # 铜 (mg)
    MAGNESIUM = 1090   # 镁 (mg)
    POTASSIUM = 1092   # 钾 (mg)
    SODIUM = 1093      # 钠 (mg)
    SELENIUM = 1103    # 硒 (μg)
    VITAMIN_A = 1104   # 维生素A (IU)
    VITAMIN_D = 1110   # 维生素D (IU)
    VITAMIN_E = 1109   # 维生素E (mg)
    VITAMIN_B1 = 1165  # 维生素B1 (mg)
    VITAMIN_B12 = 1178 # 维生素B12 (μg)
    RIBOFLAVIN = 1166  # 核黄素 (mg)
    OMEGA_6 = 1316     # ω-6脂肪酸 (g)
    OMEGA_3 = 1404     # ω-3脂肪酸 (g)

    @classmethod
    def get_all_ids(cls) -> List[int]:
        """获取所有营养素ID"""
        return [getattr(cls, attr) for attr in dir(cls) 
                if not attr.startswith('__') and not callable(getattr(cls, attr))]

    @classmethod
    def get_name(cls, nutrient_id: int) -> str:
        """根据ID获取营养素名称"""
        for attr in dir(cls):
            if not attr.startswith('__') and not callable(getattr(cls, attr)):
                if getattr(cls, attr) == nutrient_id:
                    return attr.lower().replace('_', ' ')
        return f"营养素 {nutrient_id}"
    
class FoodCategory:
    """食物类别枚举"""
    PROTEIN = 'protein'
    SUBPROTEIN = 'subprotein'
    FAT = 'fat'
    CARB = 'carb'
    VEGETABLE = 'vegetable'
    FRUIT = 'fruit'
    SUPPLEMENT = 'supplement'

    @classmethod
    def get_all_categories(cls) -> List[str]:
        """获取所有食物类别"""
        return [getattr(cls, attr) for attr in dir(cls) 
                if not attr.startswith('__') and not callable(getattr(cls, attr))]

class Config:
    """配置管理类，负责加载和保存配置"""
    
    @staticmethod
    def load_config() -> configparser.ConfigParser:
        """加载配置文件，如果不存在则创建默认配置"""
        config = configparser.ConfigParser()
        
        # 默认配置
        default_config = {
            'FoodCategoryRatio': {
                'protein_min': '0.30', 
                'protein_max': '0.40',
                'subprotein_min': '0.05', 
                'subprotein_max': '0.10',
                'fat_min': '0.05', 
                'fat_max': '0.10',
                'carb_min': '0.15', 
                'carb_max': '0.25',
                'vegetable_min': '0.10', 
                'vegetable_max': '0.20',
                'fruit_min': '0.05', 
                'fruit_max': '0.10'
            },
            'FoodCategoryCount': {
                'protein_min': '1', 
                'protein_max': '2',
                'subprotein_min': '0', 
                'subprotein_max': '1',
                'fat_min': '1', 
                'fat_max': '1',
                'carb_min': '1', 
                'carb_max': '1',
                'vegetable_min': '1', 
                'vegetable_max': '3',
                'fruit_min': '1', 
                'fruit_max': '2'
            },
            'CaToP': {
                'min': '1.0',
                'max': '2.0'
            },
            'Optimization': {
                'default_alpha': '110.0',
                'default_beta': '120.0',
                'default_variety_factor': '1.0',
                'max_food_weight': '3000',
                'min_usage': '0.1',
                'solver_time_limit': '60',
                'solver_gap_rel': '0.01'
            }
        }
        
        # 检查配置文件是否存在
        if CONFIG_PATH.exists():
            config.read(CONFIG_PATH)
        else:
            # 创建默认配置
            for section, options in default_config.items():
                config[section] = options
            
            # 保存默认配置
            with open(CONFIG_PATH, 'w') as f:
                config.write(f)
        
        return config
    
    @staticmethod
    def get_food_category_ratios() -> Dict[str, Tuple[float, float]]:
        """获取食物类别比例配置"""
        config = Config.load_config()
        categories = FoodCategory.get_all_categories()
        ratios = {}
        
        for category in categories:
            if category in ['supplement']:  # 排除不需要比例约束的类别
                continue
                
            min_key = f"{category}_min"
            max_key = f"{category}_max"
            
            if min_key in config['FoodCategoryRatio'] and max_key in config['FoodCategoryRatio']:
                ratios[category] = (
                    float(config['FoodCategoryRatio'][min_key]), 
                    float(config['FoodCategoryRatio'][max_key])
                )
        
        return ratios
    
    @staticmethod
    def get_food_category_counts() -> Dict[str, Tuple[int, int]]:
        """获取食物类别数量配置"""
        config = Config.load_config()
        categories = FoodCategory.get_all_categories()
        counts = {}
        
        for category in categories:
            if category in ['supplement']:  # 排除不需要数量约束的类别
                continue
                
            min_key = f"{category}_min"
            max_key = f"{category}_max"
            
            if min_key in config['FoodCategoryCount'] and max_key in config['FoodCategoryCount']:
                counts[category] = (
                    int(config['FoodCategoryCount'][min_key]), 
                    int(config['FoodCategoryCount'][max_key])
                )
        
        return counts
    
    @staticmethod
    def get_ca_to_p_ratio() -> Tuple[float, float]:
        """获取钙磷比例配置"""
        config = Config.load_config()
        return (
            float(config['CaToP']['min']),
            float(config['CaToP']['max'])
        )
    
    @staticmethod
    def get_optimization_params() -> Dict[str, Any]:
        """获取优化参数配置"""
        config = Config.load_config()
        return {
            'default_alpha': float(config['Optimization']['default_alpha']),
            'default_beta': float(config['Optimization']['default_beta']),
            'default_variety_factor': float(config['Optimization']['default_variety_factor']),
            'max_food_weight': float(config['Optimization']['max_food_weight']),
            'min_usage': float(config['Optimization']['min_usage']),
            'solver_time_limit': int(config['Optimization']['solver_time_limit']),
            'solver_gap_rel': float(config['Optimization']['solver_gap_rel'])
        }

# 从 data_operate.py 移动过来的函数
def create_allergy_pattern(allergy_foods: List[str]) -> Dict[str, re.Pattern]:
    """
    为每个过敏食物创建正则表达式模式
    
    参数:
    allergy_foods: List[str] - 过敏食物关键词列表
    
    返回:
    Dict[str, re.Pattern] - 过敏食物及其对应的正则表达式模式
    """
    patterns = {}
    for food in allergy_foods:
        if not food or not isinstance(food, str):
            continue
        # 创建全词匹配模式，防止部分匹配（如"apple"匹配"pineapple"）
        food_cleaned = food.lower().strip()
        pattern = r'\b' + re.escape(food_cleaned) + r'\b'
        patterns[food] = re.compile(pattern)
    return patterns

@lru_cache(maxsize=32)  # 减小缓存大小，避免内存问题
def load_food_data() -> pd.DataFrame:
    """
    加载食物数据，使用缓存避免重复读取
    
    返回:
    pd.DataFrame - 食物数据
    """
    try:
        file_path = DATA_DIR / "food_selected.csv"
        logger.info(f"Loading food data from {file_path}")
        if not file_path.exists():
            logger.error(f"Food data file not found: {file_path}")
            return pd.DataFrame(columns=['fdc_id', 'description'])
        return pd.read_csv(file_path, usecols=['fdc_id', 'description'])
    except Exception as e:
        logger.error(f"Error loading food data: {str(e)}")
        return pd.DataFrame(columns=['fdc_id', 'description'])

def find_allergic_foods(allergy_foods: List[str], min_confidence: float = 0.8) -> Dict[str, List[int]]:
    """
    在food_selected.csv中查找包含过敏食物关键词的食物ID，返回分类结果
    
    参数:
    allergy_foods: List[str] - 过敏食物关键词列表
    min_confidence: float - 最小置信度阈值，用于过滤可能的误匹配
    
    返回:
    Dict[str, List[int]] - 包含每个过敏食物对应的fdc_id列表
    """
    try:
        # 记录开始时间
        start_time = pd.Timestamp.now()
        logger.info(f"Starting allergic foods search with {len(allergy_foods)} keywords")
        
        # 参数验证
        if not allergy_foods or not isinstance(allergy_foods, list):
            logger.warning("Empty or invalid allergy foods list provided")
            return {}
        
        # 加载食物数据
        food_df = load_food_data()
        if food_df.empty:
            logger.warning("Food data is empty, cannot find allergic foods")
            return {food: [] for food in allergy_foods}
        
        # 创建正则表达式模式
        patterns = create_allergy_pattern(allergy_foods)
        
        # 初始化结果字典
        results: Dict[str, List[int]] = {food: [] for food in allergy_foods}
        
        # 将描述转换为小写
        food_df['description_lower'] = food_df['description'].str.lower()
        
        # 对每个食物描述进行匹配
        for idx, row in food_df.iterrows():
            description = row['description_lower']
            fdc_id = row['fdc_id']
            
            # 检查每个过敏食物关键词
            for food, pattern in patterns.items():
                if pattern.search(description):
                    results[food].append(fdc_id)
                    logger.debug(f"Found match: {food} in {row['description']} (ID: {fdc_id})")
        
        # 记录匹配统计
        total_matches = sum(len(ids) for ids in results.values())
        logger.info(f"Found {total_matches} total matches across {len(allergy_foods)} allergy types")
        
        # 记录处理时间
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # 记录每个过敏食物的匹配数量
        for food, ids in results.items():
            if ids:  # 只记录有匹配的食物，减少日志量
                logger.info(f"{food}: {len(ids)} matches")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in find_allergic_foods: {str(e)}")
        return {food: [] for food in allergy_foods}

def get_all_allergic_fdc_ids(allergy_foods: List[str]) -> Set[int]:
    """
    获取所有过敏食物的fdc_id集合
    
    参数:
    allergy_foods: List[str] - 过敏食物关键词列表
    
    返回:
    Set[int] - 所有过敏食物的fdc_id集合
    """
    results = find_allergic_foods(allergy_foods)
    all_ids = set()
    for ids in results.values():
        all_ids.update(ids)
    return all_ids

def _categorize_food(fdc_id: int, 
                   description: str, 
                   food_df: pd.DataFrame, 
                   has_category_label: bool, 
                   food_categories: Dict[str, List[int]]) -> None:
    """
    根据食物信息将食物分类到相应类别
    
    参数:
    fdc_id: int - 食物ID
    description: str - 食物描述
    food_df: pd.DataFrame - 食物分类信息数据框
    has_category_label: bool - 是否有食物类别标签列
    food_categories: Dict[str, List[int]] - 食物分类字典，会被修改
    
    返回:
    None - 直接修改传入的food_categories字典
    """
    # 转为小写以便匹配
    description_lower = description.lower()
    
    # 使用food_category_label列进行分类
    if has_category_label:
        food_info = food_df[food_df['fdc_id'] == fdc_id]
        if not food_info.empty:
            category_label = food_info['food_category_label'].iloc[0]
            
            if pd.notna(category_label):  # 检查不是NaN
                # 映射标签到我们的类别
                category_map = {
                    'protein': FoodCategory.PROTEIN,
                    'subprotein': FoodCategory.SUBPROTEIN,
                    'carbohydrate': FoodCategory.CARB,
                    'fat': FoodCategory.FAT,
                    'vegetable': FoodCategory.VEGETABLE,
                    'fruit': FoodCategory.FRUIT,
                    'supplement': FoodCategory.SUPPLEMENT
                }
                
                if category_label in category_map:
                    mapped_category = category_map[category_label]
                    food_categories[mapped_category].append(fdc_id)
                    return
    
    # 如果没有分类信息或分类标签不在预定义类别中，根据描述进行分类
    # 使用分类词典进行关键词匹配
    category_keywords = {
        FoodCategory.PROTEIN: ['meat', 'beef', 'chicken', 'pork', 'fish', 'turkey', 'lamb', 'veal'],
        FoodCategory.SUBPROTEIN: ['organ', 'liver', 'heart', 'kidney', 'yogurt', 'egg'],
        FoodCategory.FAT: ['oil', 'fat', 'butter', 'lard', 'tallow'],
        FoodCategory.CARB: ['rice', 'pasta', 'bread', 'potato', 'cereal', 'grain', 'flour'],
        FoodCategory.VEGETABLE: ['vegetable', 'carrot', 'broccoli', 'spinach', 'cabbage', 'lettuce'],
        FoodCategory.FRUIT: ['fruit', 'apple', 'banana', 'orange', 'berry', 'pear', 'melon'],
        FoodCategory.SUPPLEMENT: ['supplement', 'vitamin', 'mineral', 'fiber', 'probiotic']
    }
    
    # 使用正则表达式进行一次性匹配，提高性能
    for category, keywords in category_keywords.items():
        pattern = '|'.join(r'\b' + re.escape(word) + r'\b' for word in keywords)
        if re.search(pattern, description_lower):
            food_categories[category].append(fdc_id)
            return
    
    # 如果无法通过关键词分类，默认为辅助蛋白类
    food_categories[FoodCategory.SUBPROTEIN].append(fdc_id)

def get_food_dict(allergy_foods_id: Optional[List[int]] = None) -> Tuple[Dict, Dict]:
    """
    获取处理后的食物字典和食物分类，根据提供的过敏食物ID列表进行过滤
    
    参数:
    allergy_foods_id: List[int] - 需要排除的过敏食物ID列表，如果为None则不排除任何食物
    
    返回:
    tuple - (食物字典, 食物分类)，食物字典格式为 
            {fdc_id: {'id': fdc_id, 'description': description, 
                    'nutrients': {nutrient_id: {'name': name, 'amount': amount, 'unit': unit_name}}}}
    """
    food_dict = {}
    food_categories = {category: [] for category in FoodCategory.get_all_categories()}
    
    try:
        # 只读取需要的列，减少内存使用
        needed_columns = ['fdc_id', 'description', 'nutrient_id', 'name', 'amount', 'unit_name']
        
        # 读取所有食物数据，使用绝对路径
        food_data_path = DATA_DIR / "food_data.csv"
        if not food_data_path.exists():
            logger.error(f"Food data file not found: {food_data_path}")
            return food_dict, food_categories
        
        food_data_df = pd.read_csv(food_data_path, usecols=needed_columns)
        
        # 使用 query 方法进行高效筛选
        if allergy_foods_id:
            # 将列表转换为集合，提高查找效率
            excluded_ids = set(allergy_foods_id)
            # 使用 isin 和否定操作进行高效筛选，排除过敏食物
            food_data_df = food_data_df[~food_data_df['fdc_id'].isin(excluded_ids)]
            logger.info(f"Excluded {len(excluded_ids)} allergic foods from the dataset")
        
        # 如果没有数据，返回空字典
        if food_data_df.empty:
            logger.warning("No food data available after filtering")
            return food_dict, food_categories
    
        # 读取食物分类信息      
        try:
            food_selected_path = DATA_DIR / "food_selected.csv"
            if not food_selected_path.exists():
                logger.warning(f"Food selected file not found: {food_selected_path}")
                food_df = pd.DataFrame(columns=['fdc_id', 'description', 'food_category_id', 'food_category_label'])
                has_category_label = False
            else:
                food_df = pd.read_csv(food_selected_path)
                has_category_label = 'food_category_label' in food_df.columns
        except (FileNotFoundError, KeyError) as e:
            # 如果文件不存在或没有所需列，使用默认分类
            logger.warning(f"Error loading food_selected.csv: {str(e)}. Using default categorization.")
            food_df = pd.DataFrame(columns=['fdc_id', 'description', 'food_category_id', 'food_category_label'])
            has_category_label = False
        
        # 使用Pandas的向量化操作进行优化
        # 首先，按fdc_id和description分组，创建营养素字典
        grouped = food_data_df.groupby(['fdc_id', 'description'])
        
        # 遍历每个组并创建食物字典
        for (fdc_id, description), group in grouped:
            # 创建营养素字典
            nutrients_dict = {
                row['nutrient_id']: {
                    'name': row['name'],
                    'amount': row['amount'],
                    'unit': row['unit_name']
                } for _, row in group.iterrows()
            }
            
            food_dict[fdc_id] = {
                'id': fdc_id,
                'description': description,
                'nutrients': nutrients_dict
            }
            
            # 对食物进行分类
            _categorize_food(fdc_id, description, food_df, has_category_label, food_categories)
    
        logger.info(f"Created food dictionary with {len(food_dict)} foods")
        logger.info(f"Food categories: {sum(len(cat) for cat in food_categories.values())} foods categorized")
        
        return food_dict, food_categories
    
    except Exception as e:
        logger.error(f"Error in get_food_dict: {str(e)}", exc_info=True)
        # 返回空结果
        return food_dict, food_categories

def load_aafco_constraints(state: str = "adult") -> Dict[int, Dict]:
    """
    从配置加载AAFCO标准的营养素需求
    
    参数:
    state: str - 宠物生命阶段，"adult"或"child"
    
    返回:
    Dict[int, Dict] - 营养素需求字典，格式为{nutrient_id: {"min": min_value, "max": max_value, "unit": unit}}
    """
    # 成年犬营养需求(每1000千卡)
    adult_nutrient_req = {
        NutrientID.PROTEIN: {"min": 45.0, "max": None, 'unit': 'G'},
        NutrientID.FAT: {"min": 13.8, "max": None, 'unit': 'G'},
        NutrientID.ENERGY: {"min": 2000, "max": None, 'unit': 'KCAL'},
        NutrientID.CALCIUM: {"min": 1.25*1000, "max": 6.25*1000, 'unit': 'MG'},
        NutrientID.PHOSPHORUS: {"min": 1.0*1000, "max": 4.0*1000, 'unit': 'MG'},
        NutrientID.POTASSIUM: {"min": 1.5*1000, "max": None, 'unit': 'MG'},
        NutrientID.SODIUM: {"min": 0.2*1000, "max": None, 'unit': 'MG'},
        NutrientID.MAGNESIUM: {"min": 0.15*1000, "max": None, 'unit': 'MG'},
        NutrientID.IRON: {"min": 10, "max": None, 'unit': 'MG'},
        NutrientID.ZINC: {"min": 20, "max": None, 'unit': 'MG'},
        NutrientID.COPPER: {"min": 1.83, "max": None, 'unit': 'MG'},
        NutrientID.SELENIUM: {"min": 0.08*1000, "max": 0.5*1000, 'unit': 'UG'},
        NutrientID.VITAMIN_A: {"min": 1250, "max": 62500, 'unit': 'IU'},
        NutrientID.VITAMIN_E: {"min": 12.5 * 0.6774, "max": None, 'unit': 'MG'}, # 转换系数: 1IU = 0.6774 mg
        NutrientID.VITAMIN_D: {"min": 125, "max": 750, 'unit': 'IU'},
        NutrientID.VITAMIN_B1: {"min": 0.56, "max": None, 'unit': 'MG'},
        NutrientID.VITAMIN_B12: {"min": 0.007*1000, "max": None, 'unit': 'UG'},
        NutrientID.RIBOFLAVIN: {"min": 1.3, "max": None, 'unit': 'MG'},
        NutrientID.OMEGA_6: {"min": 3.3, "max": None, 'unit': 'G'}, # LA
        NutrientID.OMEGA_3: {"min": 0.2, "max": None, 'unit': 'G'}  # ALA
    }

    # 幼犬营养需求
    child_nutrient_req = {
        NutrientID.PROTEIN: {"min": 30, "max": None, 'unit': 'G'},
        NutrientID.FAT: {"min": 21.3, "max": None, 'unit': 'G'},
        NutrientID.CALCIUM: {"min": 3.0*1000, "max": 6.25*1000, 'unit': 'MG'},
        NutrientID.PHOSPHORUS: {"min": 2.5*1000, "max": 4.0*1000, 'unit': 'MG'},
        NutrientID.POTASSIUM: {"min": 1.5*1000, "max": None, 'unit': 'MG'},
        NutrientID.SODIUM: {"min": 0.8*1000, "max": None, 'unit': 'MG'},
        NutrientID.MAGNESIUM: {"min": 0.14*1000, "max": None, 'unit': 'MG'},
        NutrientID.IRON: {"min": 22, "max": None, 'unit': 'MG'},
        NutrientID.ZINC: {"min": 25, "max": None, 'unit': 'MG'},
        NutrientID.COPPER: {"min": 3.1, "max": None, 'unit': 'MG'},
        NutrientID.SELENIUM: {"min": 0.5*1000, "max": None, 'unit': 'UG'},
        NutrientID.VITAMIN_A: {"min": 1250, "max": 62500, 'unit': 'IU'},
        NutrientID.VITAMIN_D: {"min": 125, "max": 750, 'unit': 'IU'},
        NutrientID.VITAMIN_E: {"min": 12.5, "max": None, 'unit': 'MG'},
        NutrientID.VITAMIN_B1: {"min": 0.56, "max": None, 'unit': 'MG'},
        NutrientID.VITAMIN_B12: {"min": 0.007*1000, "max": None, 'unit': 'UG'},
        NutrientID.RIBOFLAVIN: {"min": 1.3, "max": None, 'unit': 'MG'},
        NutrientID.OMEGA_6: {"min": 3.3, "max": None, 'unit': 'G'},
        NutrientID.OMEGA_3: {"min": 0.2, "max": None, 'unit': 'G'}
    }

    # 根据生命阶段选择营养需求标准
    if state.lower() == "child":
        return child_nutrient_req
    else:
        return adult_nutrient_req

def recipe_optimize(nut_requirements: Optional[Dict] = None, 
                   target_energy: Optional[Union[float, str]] = None, 
                   min_variety: int = 4, 
                   max_variety: int = 8, 
                   allergy_foods_id: Optional[List[int]] = None, 
                   alpha: Optional[float] = None, 
                   beta: Optional[float] = None, 
                   variety_factor: Optional[float] = None, 
                   max_food_weight: Optional[float] = None, 
                   min_usage: Optional[float] = None,
                   log_level: str = "INFO") -> Dict:
    """
    优化宠物食谱

    参数:
    nut_requirements: Dict - 营养素需求，格式为{营养素ID: {"min": 最小值, "max": 最大值}}，
                             如果为None则使用AAFCO成年犬标准
    target_energy: float or str - 目标能量，可以是浮点数或字符串形式的数值
    min_variety: int - 最少食物种类
    max_variety: int - 最多食物种类
    allergy_foods_id: List[int] - 过敏食物ID列表（可选）
    alpha: float - 能量偏差权重，控制能量偏差在目标函数中的重要性
    beta: float - 营养素松弛变量权重，控制营养素满足度在目标函数中的重要性
    variety_factor: float - 食物多样性权重，控制食物多样性在目标函数中的重要性
    max_food_weight: float - 单个食物最大重量(克)
    min_usage: float - 食物最小使用量(克)，如果选择使用某食物，其重量不得低于此值
    log_level: str - 日志级别，可选值: "DEBUG", "INFO", "WARNING", "ERROR"
    
    返回:
    dict - 包含优化结果的字典，包括:
        - 'status': 优化状态
        - 'selected_foods': 选定的食物列表 [(食物ID, 食物描述, 用量)]
        - 'energy': {'actual': 实际能量, 'target': 目标能量, 'deviation': 偏差}
        - 'nutrients': 各营养素的满足情况
        - 'unsatisfied_nutrients': 未满足的营养素列表
        - 'model': 优化模型 (可选，用于调试)
    """
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning(f"Invalid log level: {log_level}, using INFO instead")
        numeric_level = logging.INFO
    logger.setLevel(numeric_level)

    # 获取默认配置值
    config_params = Config.get_optimization_params()
    if alpha is None:
        alpha = config_params['default_alpha']
    if beta is None:
        beta = config_params['default_beta']
    if variety_factor is None:
        variety_factor = config_params['default_variety_factor']
    if max_food_weight is None:
        max_food_weight = config_params['max_food_weight']
    if min_usage is None:
        min_usage = config_params['min_usage']

    # 使用AAFCO标准如果没有提供营养需求
    if nut_requirements is None:
        nut_requirements = load_aafco_constraints()
        logger.info("Using default AAFCO adult dog nutrition requirements")

    # 参数验证
    if not _validate_parameters(nut_requirements, target_energy, min_variety, max_variety, alpha, beta, variety_factor):
        return _create_error_result("Parameter validation failed")
    
    # 尝试将target_energy转换为整数
    target_energy = _convert_target_energy(target_energy)
    if target_energy is None:
        return _create_error_result("Invalid target energy value")
    
    # 获取食物字典和食物分类，排除过敏食物
    try:
        food_dict, food_categories = get_food_dict(allergy_foods_id=allergy_foods_id)
    except Exception as e:
        logger.error(f"Error getting food dictionary: {e}")
        return _create_error_result(f"Failed to retrieve food data: {str(e)}")
    
    if not food_dict:
        return _create_error_result("No food data available for optimization")
    
    logger.info(f"Retrieved {len(food_dict)} foods for optimization with {len(allergy_foods_id) if allergy_foods_id else 0} allergy food IDs")

    # 创建并求解LP模型
    try:
        model, variables = _create_lp_model(
            food_dict, food_categories, nut_requirements, target_energy, 
            min_variety, max_variety, alpha, beta, variety_factor,
            max_food_weight, min_usage
        )

    # 求解模型
        solve_status = _solve_model(model)
        
        # 处理结果
        if solve_status in ["Optimal", "Feasible"]:
            return _process_optimal_results(model, variables, food_dict, food_categories, nut_requirements, target_energy)
        else:
            return _create_error_result(f"No feasible solution found. Status: {solve_status}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        return _create_error_result(f"Optimization error: {str(e)}")

def _validate_parameters(nut_requirements: Dict, 
                       target_energy: Optional[Union[float, str]], 
                       min_variety: int, 
                       max_variety: int, 
                       alpha: float, 
                       beta: float, 
                       variety_factor: float) -> bool:
    """验证输入参数有效性

    参数:
    nut_requirements: Dict - 营养素需求
    target_energy: float or str - 目标能量
    min_variety: int - 最少食物种类
    max_variety: int - 最多食物种类
    alpha: float - 能量偏差权重
    beta: float - 营养素松弛变量权重
    variety_factor: float - 食物多样性权重
    
    返回:
    bool - 参数是否有效
    """

    # 检查营养需求
    if nut_requirements is None or not isinstance(nut_requirements, dict):
        logger.error("Invalid nutrition requirements")
        return False
    
    # 检查目标能量
    if target_energy is None:
        logger.error("Target energy is required")
        return False
    
    # 检查多样性参数
    if min_variety > max_variety:
        logger.error(f"Invalid variety range: min_variety ({min_variety}) > max_variety ({max_variety})")
        return False
    
    # 检查权重参数
    if alpha <= 0 or beta <= 0 or variety_factor < 0:
        logger.error(f"Invalid weight parameters: alpha={alpha}, beta={beta}, variety_factor={variety_factor}")
        return False
    
    return True

def _convert_target_energy(target_energy: Optional[Union[float, str]]) -> Optional[int]:
    """
    将目标能量转换为整数
    
    参数:
    target_energy: float or str - 目标能量
    
    返回:
    Optional[int] - 转换后的整数能量值，转换失败则返回None
    """
    if target_energy is None:
        return None
        
    try:
        return int(float(target_energy))
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting target_energy to integer: {e}")
        logger.error(f"target_energy value: {target_energy}, type: {type(target_energy)}")
        return None
    
def _create_lp_model(food_dict: Dict, 
                    food_categories: Dict, 
                    nut_requirements: Dict, 
                    target_energy: int, 
                    min_variety: int, 
                    max_variety: int, 
                    alpha: float, 
                    beta: float, 
                    variety_factor: float,
                    max_food_weight: float, 
                    min_usage: float) -> Tuple[LpProblem, Dict]:
    """
    创建线性规划模型
    
    参数:
    food_dict: Dict - 食物字典
    food_categories: Dict - 食物分类
    nut_requirements: Dict - 营养素需求
    target_energy: int - 目标能量
    min_variety: int - 最少食物种类
    max_variety: int - 最多食物种类
    alpha: float - 能量偏差权重
    beta: float - 营养素松弛变量权重
    variety_factor: float - 食物多样性权重
    max_food_weight: float - 单个食物最大重量
    min_usage: float - 食物最小使用量
    
    返回:
    Tuple[LpProblem, Dict] - 线性规划模型和变量字典
    """
    # 创建LP模型
    model = LpProblem("PetRecipe", LpMinimize)
    
    # 创建决策变量
    x_vars = {}  # 食物用量，单位为克非负连续
    y_vars = {}  # 是否选用食物，0/1 二元变量
    
    for fid, item in food_dict.items():
        x_vars[fid] = LpVariable(f"x_{fid}", lowBound=0, cat='Continuous')
        y_vars[fid] = LpVariable(f"y_{fid}", cat='Binary')
    
    # 营养素松弛变量，对每个营养素设置slack var（s_nut >= 0）
    slack_vars = {}
    for nut in nut_requirements:
        slack_vars[nut] = LpVariable(f"slack_{nut}", lowBound=0, cat='Continuous')
    
    # 设置能量差变量
    epos = LpVariable("energy_pos_diff", lowBound=0, cat='Continuous')  # 正偏差
    eneg = LpVariable("energy_neg_diff", lowBound=0, cat='Continuous')  # 负偏差
    
    # 计算总重量表达式
    total_weight = lpSum(x_vars[fid] for fid in food_dict)
    
    # 添加各种约束
    _add_food_weight_constraints(model, x_vars, y_vars, food_dict, max_food_weight, min_usage)
    _add_food_category_constraints(model, x_vars, y_vars, food_categories, total_weight)
    _add_nutrient_constraints(model, x_vars, y_vars, slack_vars, food_dict, nut_requirements, target_energy)
    _add_energy_constraint(model, x_vars, food_dict, target_energy, epos, eneg)
    _add_variety_constraints(model, y_vars, food_dict, min_variety, max_variety)
    
    # 添加特殊约束 - 钙磷比例约束
    _add_calcium_phosphorus_ratio_constraint(model, x_vars, food_dict)
    
    # 设置目标函数：最小化能量偏差和营养素松弛变量
    slack_expr = lpSum(slack_vars.values())
    model += (alpha * (epos + eneg) + beta * slack_expr + 
              variety_factor * (max_variety - lpSum(y_vars.values()))), "Minimize_Energy_Deviation"
    
    # 返回模型和各种变量
    variables = {
        'x_vars': x_vars,
        'y_vars': y_vars,
        'slack_vars': slack_vars,
        'epos': epos,
        'eneg': eneg
    }
    
    return model, variables

def _add_food_weight_constraints(model: LpProblem, 
                                x_vars: Dict, 
                                y_vars: Dict, 
                                food_dict: Dict, 
                                max_food_weight: float, 
                                min_usage: float) -> None:
    """
    添加食物重量相关的约束
    
    参数:
    model: LpProblem - 线性规划模型
    x_vars: Dict - 食物用量变量
    y_vars: Dict - 食物选择变量
    food_dict: Dict - 食物字典
    max_food_weight: float - 单个食物最大重量
    min_usage: float - 食物最小使用量
    
    返回:
    None - 直接修改传入的模型
    """
    # 大M约束，动态设置为食物最大重量的10倍
    bigM = max_food_weight * 10
    
    # 添加每个食物的约束
    for fid in food_dict:
        # 如果选择使用某食物(y=1)，其用量受max_food_weight限制；如果不使用(y=0)，其用量为0
        model += x_vars[fid] <= max_food_weight * y_vars[fid], f"Max_Weight_{fid}"
        
        # 如果选择使用某食物(y=1)，其用量不得低于min_usage；如果不使用(y=0)，其用量为0
        model += x_vars[fid] >= min_usage * y_vars[fid], f"Min_Usage_{fid}"

def _add_food_category_constraints(model: LpProblem, 
                                  x_vars: Dict, 
                                  y_vars: Dict, 
                                  food_categories: Dict, 
                                  total_weight: float) -> None:
    """
    添加食物类别相关的约束
    
    参数:
    model: LpProblem - 线性规划模型
    x_vars: Dict - 食物用量变量
    y_vars: Dict - 食物选择变量
    food_categories: Dict - 食物分类
    total_weight: float - 总重量表达式
    
    返回:
    None - 直接修改传入的模型
    """
    # 获取配置的食物类别比例和数量约束
    category_ratios = Config.get_food_category_ratios()
    category_counts = Config.get_food_category_counts()
    
    # 为每个类别添加约束
    for category in FoodCategory.get_all_categories():
        if category == FoodCategory.SUPPLEMENT:  # 排除不需要约束的类别
            continue
            
        if category in category_ratios and category in category_counts:
            ratio_range = category_ratios[category]
            count_range = category_counts[category]
            
            # 调用通用函数添加约束
            _add_category_constraint(model, x_vars, y_vars, food_categories, category, 
                                    total_weight, count_range, ratio_range)
    
def _add_category_constraint(model: LpProblem, 
                           x_vars: Dict, 
                           y_vars: Dict, 
                           food_categories: Dict, 
                           category: str, 
                           total_weight: float, 
                           count_range: Tuple[int, int], 
                           weight_ratio: Tuple[float, float]) -> None:
    """
    为特定食物类别添加约束
    
    参数:
    model: LpProblem - 线性规划模型
    x_vars: Dict - 食物用量变量
    y_vars: Dict - 食物选择变量
    food_categories: Dict - 食物分类
    category: str - 食物类别
    total_weight: float - 总重量表达式
    count_range: Tuple[int, int] - 食物种类数范围
    weight_ratio: Tuple[float, float] - 重量比例范围
    
    返回:
    None - 直接修改传入的模型
    """
    # 确保该类别存在食物
    if category in food_categories and food_categories[category]:
        # 食物种类数量约束
        min_count, max_count = count_range
        
        # 最小种类数约束
        if min_count > 0:
            model += lpSum(y_vars[fid] for fid in food_categories[category]) >= min_count, f"Min_{category}_Count"
        
        # 最大种类数约束
        model += lpSum(y_vars[fid] for fid in food_categories[category]) <= max_count, f"Max_{category}_Count"
        
        # 重量比例约束
        min_ratio, max_ratio = weight_ratio
        
        # 如果有最小比例要求
        if min_ratio > 0:
            model += lpSum(x_vars[fid] for fid in food_categories[category]) >= min_ratio * total_weight, f"Min_{category}_Weight_Ratio"
        
        # 如果有最大比例要求
        model += lpSum(x_vars[fid] for fid in food_categories[category]) <= max_ratio * total_weight, f"Max_{category}_Weight_Ratio"

def _add_nutrient_constraints(model: LpProblem, 
                             x_vars: Dict, 
                             y_vars: Dict, 
                             slack_vars: Dict, 
                             food_dict: Dict, 
                             nut_requirements: Dict, 
                             target_energy: int) -> None:
    """
    添加营养素相关的约束
    
    参数:
    model: LpProblem - 线性规划模型
    x_vars: Dict - 食物用量变量
    y_vars: Dict - 食物选择变量
    slack_vars: Dict - 松弛变量
    food_dict: Dict - 食物字典
    nut_requirements: Dict - 营养素需求
    target_energy: int - 目标能量
    
    返回:
    None - 直接修改传入的模型
    """
    # 为每个营养素设置上下限约束
    for nutrient_id, bounds in nut_requirements.items():
        min_val = bounds.get("min", None)
        max_val = bounds.get("max", None)
        
        # 将营养素需求按照能量进行缩放
        if min_val is not None:
            min_val = min_val * target_energy / 1000
        if max_val is not None:
            max_val = max_val * target_energy / 1000
        
        # 计算营养素的总摄入量表达式
        nut_expr = _calculate_nutrient_expr(x_vars, food_dict, nutrient_id)
        
        # 设置下限约束
        if min_val is not None:
            model += (nut_expr + slack_vars[nutrient_id] >= min_val), f"Min_{nutrient_id}"
        
        # 设置上限约束
        if max_val is not None:
            model += (nut_expr <= max_val), f"Max_{nutrient_id}"

def _add_energy_constraint(model: LpProblem, 
                          x_vars: Dict, 
                          food_dict: Dict, 
                          target_energy: int, 
                          epos: LpVariable, 
                          eneg: LpVariable) -> None:
    """
    添加能量相关的约束
    
    参数:
    model: LpProblem - 线性规划模型
    x_vars: Dict - 食物用量变量
    food_dict: Dict - 食物字典
    target_energy: int - 目标能量
    epos: LpVariable - 能量正偏差变量
    eneg: LpVariable - 能量负偏差变量
    
    返回:
    None - 直接修改传入的模型
    """
    # 计算总能量表达式
    total_energy_expr = _calculate_nutrient_expr(x_vars, food_dict, NutrientID.ENERGY)
    
    # 能量平衡约束：total_energy - target_energy = epos - eneg
    model += (total_energy_expr - target_energy) == (epos - eneg), "Energy_Balance"

def _add_variety_constraints(model: LpProblem, 
                            y_vars: Dict, 
                            food_dict: Dict, 
                            min_variety: int, 
                            max_variety: int) -> None:
    """
    添加食物多样性相关的约束
    
    参数:
    model: LpProblem - 线性规划模型
    y_vars: Dict - 食物选择变量
    food_dict: Dict - 食物字典
    min_variety: int - 最少食物种类
    max_variety: int - 最多食物种类
    
    返回:
    None - 直接修改传入的模型
    """
    # 最小食物种类约束
    model += lpSum(y_vars[fid] for fid in food_dict) >= min_variety, "Minimum_Variety"
    
    # 最大食物种类约束
    model += lpSum(y_vars[fid] for fid in food_dict) <= max_variety, "Maximum_Variety"

def _add_calcium_phosphorus_ratio_constraint(model: LpProblem, 
                                            x_vars: Dict, 
                                            food_dict: Dict) -> None:
    """
    添加钙磷比例约束
    
    参数:
    model: LpProblem - 线性规划模型
    x_vars: Dict - 食物用量变量
    food_dict: Dict - 食物字典
    
    返回:
    None - 直接修改传入的模型
    """
    # 检查是否包含钙和磷的营养数据
    has_calcium = any(NutrientID.CALCIUM in food_dict[food]['nutrients'] for food in food_dict)
    has_phosphorus = any(NutrientID.PHOSPHORUS in food_dict[food]['nutrients'] for food in food_dict)
    
    if has_calcium and has_phosphorus:
        # 获取配置的钙磷比例
        ca_to_p_min, ca_to_p_max = Config.get_ca_to_p_ratio()

        # 计算钙和磷的摄入表达式
        calcium_expr = _calculate_nutrient_expr(x_vars, food_dict, NutrientID.CALCIUM)
        phosphorus_expr = _calculate_nutrient_expr(x_vars, food_dict, NutrientID.PHOSPHORUS)
        
        # 添加钙磷比例约束
        model += phosphorus_expr >= 0.1, "Min_Phosphorus"  # 添加一个小的下限防止除零
        model += calcium_expr >= ca_to_p_min * phosphorus_expr, "Min_Ca_P_Ratio"
        model += calcium_expr <= ca_to_p_max * phosphorus_expr, "Max_Ca_P_Ratio"

def _calculate_nutrient_expr(x_vars: Dict, 
                            food_dict: Dict, 
                            nutrient_id: int) -> float:
    """
    计算特定营养素的总摄入量表达式
    
    参数:
    x_vars: Dict - 食物用量变量
    food_dict: Dict - 食物字典
    nutrient_id: int - 营养素ID
    
    返回:
    float - 营养素总摄入量表达式
    """
    nutrient_expr = lpSum(
        food_dict[food]['nutrients'].get(nutrient_id, {'amount': 0})['amount'] * x_vars[food] / 100 
        for food in food_dict 
        if nutrient_id in food_dict[food]['nutrients']
    )
    return nutrient_expr

def _solve_model(model: LpProblem, time_limit: int = 150, gap_rel: float = 0.01) -> str:
    """
    求解线性规划模型
    
    参数:
    model: LpProblem - 线性规划模型
    time_limit: int - 求解时间限制(秒)
    gap_rel: float - 相对间隙容忍度
    
    返回:
    str - 求解状态
    """
    # 设置求解器参数
    solver_params = {
        'msg': 0,  # 0表示不显示求解过程，1表示显示
        'timeLimit': time_limit,  # 求解时间限制(秒)
        'gapRel': gap_rel,  # 相对间隙容忍度
    }
    
    # 求解模型
    solver = PULP_CBC_CMD(**solver_params)
    model.solve(solver)
    
    # 返回求解状态
    return LpStatus[model.status]

def _process_optimal_results(model: LpProblem, 
                            variables: Dict, 
                            food_dict: Dict, 
                            food_categories: Dict, 
                            nut_requirements: Dict, 
                            target_energy: int) -> Dict:
    """
    处理优化结果
    
    参数:
    model: LpProblem - 线性规划模型
    variables: Dict - 变量字典
    food_dict: Dict - 食物字典
    food_categories: Dict - 食物分类
    nut_requirements: Dict - 营养素需求
    target_energy: int - 目标能量
    
    返回:
    Dict - 优化结果字典
    """
    # 提取变量
    x_vars = variables['x_vars']
    y_vars = variables['y_vars']
    slack_vars = variables['slack_vars']
    epos = variables['epos']
    eneg = variables['eneg']
    
    # 筛选出被选中的食物
    selected_foods = []
    for fid in y_vars:
        if y_vars[fid].varValue > 0.5:  # 二进制变量，>0.5 视为选中
            x_val = x_vars[fid].varValue or 0
            selected_foods.append((
                fid, 
                food_dict[fid]['description'], 
                round(x_val, 2)
            ))
    
    # 计算实际能量和偏差
    actual_energy = sum(food_dict[food[0]]['nutrients'][NutrientID.ENERGY]['amount'] * food[2] / 100 for food in selected_foods)
    energy_deviation = (epos.varValue or 0) + (eneg.varValue or 0)
    logger.info(f"实际能量: {actual_energy}, 能量偏差: {energy_deviation}")
    
    # 计算营养素满足情况
    nutrients_info = {}
    for nutrient_id, bounds in nut_requirements.items():
        min_val = bounds.get("min", None)
        if min_val is not None:
            min_val = min_val * target_energy / 1000
            actual = sum(
                food_dict[food[0]]['nutrients'].get(nutrient_id, {'amount': 0})['amount'] * food[2] / 100 
                for food in selected_foods
            )
            slack = slack_vars[nutrient_id].varValue or 0
            satisfaction = ((actual - slack) / min_val * 100) if min_val > 0 else 100
            
            # 获取营养素名称和单位
            nut_name = NutrientID.get_name(nutrient_id)
            nut_unit = bounds.get('unit', '')
            
            for food in food_dict.values():
                if nutrient_id in food['nutrients']:
                    nut_name = food['nutrients'][nutrient_id]['name']
                    nut_unit = food['nutrients'][nutrient_id]['unit']
                    break
            
            nutrients_info[nutrient_id] = {
                'id': nutrient_id,
                'name': nut_name,
                'unit': nut_unit,
                'actual': round(actual, 2),
                'required': round(min_val, 2),
                'satisfaction': round(satisfaction, 2),
                'slack': round(slack, 2)
            }
    
    # 记录未满足的营养素
    unsatisfied_nutrients = [
        f"{nutrients_info[nutrient_id]['name']} (ID={nutrient_id}): {nutrients_info[nutrient_id]['slack']:.2f} {nutrients_info[nutrient_id]['unit']}" 
        for nutrient_id in nutrients_info 
        if nutrients_info[nutrient_id]['slack'] > 0.01
    ]
    
    if unsatisfied_nutrients:
        logger.warning("Unsatisfied nutrients: " + ", ".join(unsatisfied_nutrients))
    
    # 构建返回结果
    result = {
        'status': LpStatus[model.status],
        'selected_foods': selected_foods,
        'energy': {
            'actual': round(actual_energy, 2),
            'target': target_energy,
            'deviation': round(energy_deviation, 2)
        },
        'nutrients': nutrients_info,
        'unsatisfied_nutrients': unsatisfied_nutrients,
        'categories': _summarize_category_distribution(selected_foods, food_dict, food_categories),
        'model': model  # 可选，用于调试
    }
    logger.info(f"选中的食物: {selected_foods}")
    
    return result

def _summarize_category_distribution(selected_foods: List[Tuple[int, str, float]], 
                                food_dict: Dict, 
                                food_categories: Dict) -> Dict[str, Dict]:
    """
    统计所选食物的类别分布
    
    参数:
    selected_foods: List[Tuple] - 选定的食物列表 [(食物ID, 食物描述, 用量)]
    food_dict: Dict - 食物字典
    food_categories: Dict - 食物分类
    
    返回:
    Dict[str, Dict] - 类别分布统计
    """
    # 初始化结果
    category_stats = {}
    
    # 计算总重量
    total_weight = sum(food[2] for food in selected_foods)
    
    # 为每个类别计算统计信息
    for category, food_ids in food_categories.items():
        # 该类别选中的食物
        category_foods = [(food[0], food[1], food[2]) for food in selected_foods if food[0] in food_ids]
        
        if category_foods:
            # 计算该类别的总重量
            category_weight = sum(food[2] for food in category_foods)
            
            # 计算该类别占总重量的百分比
            percentage = (category_weight / total_weight * 100) if total_weight > 0 else 0
            
            # 记录统计信息
            category_stats[category] = {
                'foods': category_foods,
                'count': len(category_foods),
                'weight': round(category_weight, 2),
                'percentage': round(percentage, 2)
            }
    
    return category_stats

def _create_error_result(error_message: str) -> Dict:
    """
    创建错误结果
    
    参数:
    error_message: str - 错误消息
    
    返回:
    Dict - 错误结果字典
    """
    logger.error(error_message)
    return {
        'status': 'Error',
        'error_message': error_message,
        'selected_foods': [],
        'energy': {'actual': 0, 'target': 0, 'deviation': 0},
        'nutrients': {},
        'unsatisfied_nutrients': [],
        'categories': {}
    }
def visualize_solution(result: Dict, show_plots: bool = True, save_path: Optional[str] = None) -> None:
    """
    可视化优化结果（用于调试和结果展示）
    
    参数:
    result: Dict - 优化结果
    show_plots: bool - 是否显示图表
    save_path: str - 保存图表的路径，如果为None则不保存
    
    返回:
    None - 显示或保存图表
    """
    if result['status'] != 'Optimal' and result['status'] != 'Feasible':
        print(f"无法可视化结果，状态: {result['status']}")
        if 'error_message' in result:
            print(f"错误信息: {result['error_message']}")
        return
    
    print("\n===== 宠物食谱优化结果 =====")
    print(f"优化状态: {result['status']}")
    print(f"能量: 实际={result['energy']['actual']}kcal, 目标={result['energy']['target']}kcal, 偏差={result['energy']['deviation']}kcal")
    
    print("\n选定食物:")
    for fid, desc, amount in result['selected_foods']:
        print(f"  - {desc}: {amount}g")
    
    print("\n营养素满足情况:")
    for nutrient_id, info in result['nutrients'].items():
        print(f"  - {info['name']}: {info['actual']}/{info['required']} ({info['satisfaction']}%)")
    
    if result['unsatisfied_nutrients']:
        print("\n未满足的营养素:")
        for nut in result['unsatisfied_nutrients']:
            print(f"  - {nut}")
    
    if show_plots or save_path:
        try:
            _create_visualization_plots(result, show_plots, save_path)
        except Exception as e:
            logger.error(f"Error creating visualization plots: {e}", exc_info=True)
            print(f"创建可视化图表时出错: {str(e)}")

def _create_visualization_plots(result: Dict, show_plots: bool = True, save_path: Optional[str] = None) -> None:
    """
    创建结果可视化图表
    
    参数:
    result: Dict - 优化结果
    show_plots: bool - 是否显示图表
    save_path: str - 保存图表的路径，如果为None则不保存
    
    返回:
    None - 显示或保存图表
    """
    # 创建一个2x2的图表布局
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 食物分类饼图
    _plot_food_category_pie(axs[0, 0], result['categories'])
    
    # 2. 营养素满足度条形图
    _plot_nutrient_satisfaction_bar(axs[0, 1], result['nutrients'])
    
    # 3. 各食物重量柱状图
    _plot_food_weights_bar(axs[1, 0], result['selected_foods'])
    
    # 4. 实际vs目标能量
    _plot_energy_comparison(axs[1, 1], result['energy'])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图表
    if show_plots:
        plt.show()
    else:
        plt.close()

def _plot_food_category_pie(ax: plt.Axes, categories: Dict[str, Dict]) -> None:
    """绘制食物分类饼图"""
    # 提取数据
    labels = [cat.capitalize() for cat in categories.keys()]
    sizes = [info['percentage'] for info in categories.values()]
    
    # 创建饼图
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # 设置标题和一些属性
    ax.set_title('食物类别分布', fontsize=14)
    ax.axis('equal')  # 保持圆形
    
    # 设置文本属性
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_weight('bold')

def _plot_nutrient_satisfaction_bar(ax: plt.Axes, nutrients: Dict[int, Dict]) -> None:
    """绘制营养素满足度条形图"""
    # 提取数据
    labels = [info['name'] for info in nutrients.values()]
    # 直接使用浮点数，不需要strip操作
    satisfaction = [float(info['satisfaction']) for info in nutrients.values()]
    
    # 截取名称长度，避免重叠
    labels = [label[:20] + '...' if len(label) > 20 else label for label in labels]
    
    # 创建条形图
    bars = ax.barh(labels, satisfaction, color='skyblue')
    
    # 添加垂直线表示100%满足度
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
    ax.set_title('营养素满足度', fontsize=14)
    ax.set_xlabel('满足度百分比')
    
    # 设置x轴范围，确保能看到所有数据
    ax.set_xlim(0, max(150, max(satisfaction) * 1.1))
    
    # 调整布局
    ax.grid(axis='x', linestyle='--', alpha=0.7)

def _plot_food_weights_bar(ax: plt.Axes, selected_foods: List[Tuple[int, str, float]]) -> None:
    """绘制各食物重量柱状图"""
    # 提取数据
    labels = [food[1] for food in selected_foods]
    weights = [food[2] for food in selected_foods]
    
    # 截取名称长度，避免重叠
    labels = [label[:20] + '...' if len(label) > 20 else label for label in labels]
    
    # 创建条形图
    bars = ax.barh(labels, weights, color='lightgreen')
    
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
    ax.set_title('各食物重量', fontsize=14)
    ax.set_xlabel('重量 (克)')
    
    # 调整布局
    ax.grid(axis='x', linestyle='--', alpha=0.7)

def _plot_energy_comparison(ax: plt.Axes, energy: Dict[str, float]) -> None:
    """绘制实际vs目标能量比较图"""
    # 提取数据
    labels = ['目标能量', '实际能量']
    values = [energy['target'], energy['actual']]
    
    # 创建条形图
    bars = ax.bar(labels, values, color=['lightblue', 'lightgreen'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f'{height:.1f} kcal', 
            ha='center', 
            va='bottom'
        )
    
    # 设置标题
    ax.set_title('能量对比', fontsize=14)
    ax.set_ylabel('能量 (kcal)')
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加偏差说明
    deviation_text = f'偏差: {energy["deviation"]:.2f} kcal ({energy["deviation"]/energy["target"]*100:.2f}%)'
    ax.text(0.5, 0.02, deviation_text, transform=ax.transAxes, ha='center', fontsize=12)

def convert_numpy(obj):
    """转换NumPy数据类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(key): convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def export_recipe_to_file(result: Dict, file_path: str, format_type: str = 'txt') -> bool:
    """
    将食谱结果导出到文件
    
    参数:
    result: Dict - 优化结果
    file_path: str - 文件保存路径
    format_type: str - 文件格式，支持'txt', 'json'
    
    返回:
    bool - 是否导出成功
    """
    try:
        if format_type.lower() == 'json':
            # 创建可序列化的结果副本
            serializable_result = result.copy()
            # 移除不可序列化的模型对象
            if 'model' in serializable_result:
                del serializable_result['model']
            
            # 转换所有 NumPy 数据类型为 Python 原生类型
            serializable_result = convert_numpy(serializable_result)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
                
        elif format_type.lower() == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("===== 宠物食谱优化结果 =====\n")
                f.write(f"优化状态: {result['status']}\n")
                f.write(f"能量: 实际={result['energy']['actual']}kcal, 目标={result['energy']['target']}kcal, 偏差={result['energy']['deviation']}kcal\n\n")
                
                f.write("选定食物:\n")
                for fid, desc, amount in result['selected_foods']:
                    f.write(f"  - {desc}: {amount}g\n")
                
                f.write("\n营养素满足情况:\n")
                for nutrient_id, info in result['nutrients'].items():
                    f.write(f"  - {info['name']}: {info['actual']}/{info['required']} ({info['satisfaction']}%)\n")
                
                if result['unsatisfied_nutrients']:
                    f.write("\n未满足的营养素:\n")
                    for nut in result['unsatisfied_nutrients']:
                        f.write(f"  - {nut}\n")
                        
                f.write("\n食物类别分布:\n")
                for category, info in result.get('categories', {}).items():
                    f.write(f"  - {category.capitalize()}: {info['count']}种, {info['weight']}g ({info['percentage']}%)\n")
        else:
            logger.error(f"Unsupported export format: {format_type}")
            return False
            
        logger.info(f"Recipe exported successfully to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting recipe to file: {e}", exc_info=True)
        return False

def example_usage(plot_result: bool = True, export_result: bool = False) -> Dict:
    """
    示例用法，展示如何使用宠物食谱优化函数
    
    参数:
    plot_result: bool - 是否绘制结果图表
    export_result: bool - 是否导出结果到文件
    
    返回:
    Dict - 优化结果
    """
    # 定义营养素需求（使用默认的AAFCO标准）
    nut_requirements = load_aafco_constraints("adult")
    
    # 定义过敏食物
    allergy_foods = [
        'pork', 'chocolate', 'coffee', 'grape', 'raisin', 'onion', 'garlic',
        'alcohol', 'avocado', 'mushroom', 'macadamia', 'raw meat'
    ]
    
    # 获取过敏食物ID
    try:
        allergy_food_ids = list(get_all_allergic_fdc_ids(allergy_foods))
        logger.info(f"Found {len(allergy_food_ids)} food IDs matching allergy keywords")
    except Exception as e:
        logger.error(f"Error finding allergic foods: {e}")
        allergy_food_ids = []
    
    # 调用优化函数
    result = recipe_optimize(
        nut_requirements=nut_requirements,
        target_energy=1500,  # 目标能量(kcal)
        min_variety=4,       # 最少食物种类
        max_variety=8,       # 最多食物种类
        allergy_foods_id=allergy_food_ids,  # 过敏食物ID列表
        alpha=110.0,         # 能量偏差权重
        beta=120.0,           # 营养素松弛变量权重
        variety_factor=1.0,  # 食物多样性权重
        log_level="INFO"     # 日志级别
    )
    
    # 可视化结果
    if plot_result and result['status'] in ['Optimal', 'Feasible']:
        try:
            # 创建输出目录
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # 可视化结果
            visualize_solution(
                result, 
                show_plots=True, 
                save_path=str(output_dir / 'recipe_visualization.png')
            )
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
    
    # 导出结果
    if export_result and result['status'] in ['Optimal', 'Feasible']:
        try:
            # 创建输出目录
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # 导出为JSON和文本格式
            export_recipe_to_file(result, str(output_dir / 'recipe_result.json'), 'json')
            export_recipe_to_file(result, str(output_dir / 'recipe_result.txt'), 'txt')
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
    
    return result

def main():
    """主函数，用于命令行调用"""
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='宠物食谱优化工具')
    parser.add_argument('--energy', type=float, required=True, help='目标能量(kcal)')
    parser.add_argument('--min-variety', type=int, default=4, help='最少食物种类')
    parser.add_argument('--max-variety', type=int, default=8, help='最多食物种类')
    parser.add_argument('--allergy', type=str, nargs='*', default=[], help='过敏食物关键词列表')
    parser.add_argument('--stage', type=str, choices=['adult', 'child'], default='adult', help='宠物生命阶段')
    parser.add_argument('--alpha', type=float, default=110.0, help='能量偏差权重')
    parser.add_argument('--beta', type=float, default=120.0, help='营养素松弛变量权重')
    parser.add_argument('--variety-factor', type=float, default=1.0, help='食物多样性权重')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--plot', action='store_true', help='绘制结果图表')
    parser.add_argument('--export', action='store_true', help='导出结果到文件')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='日志级别')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 获取过敏食物ID
    allergy_food_ids = list(get_all_allergic_fdc_ids(args.allergy)) if args.allergy else []
    
    # 加载营养需求
    nut_requirements = load_aafco_constraints(args.stage)
    logger.info(f"加载营养需求: {nut_requirements}")
    
    # 调用优化函数
    result = recipe_optimize(
        nut_requirements=nut_requirements,
        target_energy=args.energy,
        min_variety=args.min_variety,
        max_variety=args.max_variety,
        allergy_foods_id=allergy_food_ids,
        alpha=args.alpha,
        beta=args.beta,
        variety_factor=args.variety_factor,
        log_level=args.log_level
    )
    
    # 可视化和导出结果
    if result['status'] in ['Optimal', 'Feasible']:
        # 可视化结果
        if args.plot:
            visualize_solution(result, True, str(output_dir / 'recipe_visualization.png'))
            logger.info(f"可视化结果已保存到 {output_dir / 'recipe_visualization.png'}")
        
        # 导出结果
        if args.export:
            export_recipe_to_file(result, str(output_dir / 'recipe_result.json'), 'json')
            export_recipe_to_file(result, str(output_dir / 'recipe_result.txt'), 'txt')
            logger.info(f"优化结果已保存到 {output_dir / 'recipe_result.json'} 和 {output_dir / 'recipe_result.txt'}")
    else:
        print(f"优化失败: {result.get('error_message', '未知错误')}")
    
    return result

if __name__ == '__main__':
    # 通过命令行调用主函数
    main()
