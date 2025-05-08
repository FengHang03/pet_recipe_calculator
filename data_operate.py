"""
数据处理工具模块 - 食物数据处理和过敏食物检测

本模块提供了用于处理食物数据和检测过敏食物的工具函数。

作者：
日期：2025-04-27
版本：1.0.0
"""

import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Union, Any
import logging
import os
import re
from functools import lru_cache
from pathlib import Path


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取项目根目录，确保文件路径一致性
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"

# 确保数据目录存在
DATA_DIR.mkdir(exist_ok=True)

target_nutrients = [1003, 1004, 1005, 1008, 1087, 1088, 1089, 1090, 1091, 
                        1092, 1093, 1095, 1098, 1100, 1101, 1103, 1104, 1110, 
                        1109, 1165, 1166, 1167, 1170, 1171, 1178, 1180, 1186, 
                        1210, 1211, 1212, 1213, 1214, 1215, 1217, 1219, 1220, 
                        1221, 1229, 1230, 1272, 1278, 1316, 1404]
protein_target = [173847, 173844, 175298, 175085, 172603, 175304, 174346, 172544, 174422, 174400,  # Lamb, Veal, and Game products
                  175265, 172616, 174879, 174426, 
                  # Finfish and ShellFish Products
                  175178, 172004, 175168, 175177, 171989, 175180, 174201, 174217, 171975, 174246, 
                  175137, 174238, 173679, 171994, 174235, 173707,
                  # Poultry Products
                  169902, 171140, 172853, 171492, 169905, 169903, 171511, 171494, 174494, 171117, 
                  # Beef products
                  171791, 174032, 
                  # yogurt
                  170886, 171304, 171259]
subprotein_target = [# yogurt
                    170886, 171304, 171259,
                    # Organ beef
                    169448, 173081, 168626, 
                    # Poultry
                    171485, 171059, 171061, 171487, 171457, 171483,  
                    # lamb
                    172532, 172528]
vegetable_target = [168394, 169211, 169967, 169969, 169976, 168568, 170394, 170397, 169390, 169989, 
                    170407, 169225, 168413, 168430, 168433, 170438, 170440, 
                    168463, 168486, 170475, 169302, 169292, 169283]
fruits_target = [171688, 169092, 169911, 168153, 169914, 171711, 171722, 168151, 169910,
                 169926, 168177, 169118, 169124, 169949, 167762, 173946, 167765, 169097, 169928]
fat_target = [172344, 172343, 173578, 172336, 171412, 171413, 171016, 171429, 171030, 170571,
              170158, 170183, 169414, 170563, 170557, 167702]
carb_target = [170490, 168449, 168539, 169999, 168484, 170072, 173728, 173735, 173740, 175194, 
               173753, 173757, 175211, 174257, 172429, 168395, 170420, 170285, 168917, 168871, 
               168875, 169704, 173263, 168882, 168880, 169711]
supplement_target = []

def create_allergy_pattern(allergy_foods: List[str]) -> Dict[str, re.Pattern]:
    """
    为每个过敏食物创建正则表达式模式，使用全词匹配防止误匹配
    
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
                logger.debug(f"{food}: {len(ids)} matches")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in find_allergic_foods: {str(e)}", exc_info=True)
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
                    'protein': 'protein',
                    'subprotein': 'subprotein',
                    'carbohydrate': 'carb',
                    'fat': 'fat',
                    'vegetable': 'vegetable',
                    'fruit': 'fruit',
                    'supplement': 'supplement'
                }
                
                if category_label in category_map:
                    mapped_category = category_map[category_label]
                    food_categories[mapped_category].append(fdc_id)
                    return
    
    # 如果没有分类信息或分类标签不在预定义类别中，根据描述进行分类
    # 使用分类词典进行关键词匹配
    category_keywords = {
        'protein': ['meat', 'beef', 'chicken', 'pork', 'fish', 'turkey', 'lamb', 'veal'],
        'subprotein': ['organ', 'liver', 'heart', 'kidney', 'yogurt', 'egg'],
        'fat': ['oil', 'fat', 'butter', 'lard', 'tallow'],
        'carb': ['rice', 'pasta', 'bread', 'potato', 'cereal', 'grain', 'flour'],
        'vegetable': ['vegetable', 'carrot', 'broccoli', 'spinach', 'cabbage', 'lettuce'],
        'fruit': ['fruit', 'apple', 'banana', 'orange', 'berry', 'pear', 'melon'],
        'supplement': ['supplement', 'vitamin', 'mineral', 'fiber', 'probiotic']
    }
    
    # 使用正则表达式进行一次性匹配，提高性能
    for category, keywords in category_keywords.items():
        pattern = '|'.join(r'\b' + re.escape(word) + r'\b' for word in keywords)
        if re.search(pattern, description_lower):
            food_categories[category].append(fdc_id)
            return
    
    # 如果无法通过关键词分类，默认为辅助蛋白类
    food_categories['subprotein'].append(fdc_id)


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
    # 初始化结果字典
    food_dict = {}
    food_categories = {
        'protein': [],    # 蛋白类
        'subprotein': [], # 辅助蛋白类（包括器官类）
        'carb': [],      # 碳水类
        'fat': [],       # 脂肪类
        'vegetable': [], # 蔬菜类
        'fruit': [],     # 水果类
        'supplement': [] # 补充剂类
    }
    
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
        # 返回空结果而不是抛出异常，使函数更健壮
        return food_dict, food_categories


def merge_food_data(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    合并食物数据，生成food_data.csv文件
    
    参数:
    output_path: str - 输出文件路径，如果为None则使用默认路径
    
    返回:
    pd.DataFrame - 合并后的数据框
    """
    try:
        # 确定输出路径
        if output_path is None:
            output_path = DATA_DIR / "food_data.csv"
        else:
            output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(exist_ok=True)
        
        # 定义需要排除的描述关键词
        description_contains = ['canned', 'fried', 'with salt', 'juice', 'smoked', 
                               'dried', 'frozen', 'uncooked', 'seeds']
        pattern = '|'.join(description_contains)
        
        # 读取food.csv
        food_path = DATA_DIR / "food.csv"
        if not food_path.exists():
            logger.error(f"Food data file not found: {food_path}")
            return pd.DataFrame()
            
        # 读取并过滤food.csv
        logger.info(f"Reading food data from {food_path}")
        food_df = pd.read_csv(food_path, usecols=['fdc_id', 'description', 'food_category_id'])
        
        # 过滤包含特定关键词的描述
        food_df = food_df[~food_df['description'].str.contains(pattern, case=False, na=False)]
        
        # 定义目标营养素ID
        target_nutrients = [1003, 1004, 1005, 1008, 1087, 1088, 1089, 1090, 1091, 
                           1092, 1093, 1095, 1098, 1100, 1101, 1103, 1104, 1110, 
                           1109, 1165, 1166, 1167, 1170, 1171, 1178, 1180, 1186]
        
        # 读取food_nutrient.csv
        food_nutrient_path = DATA_DIR / "food_nutrient.csv"
        if not food_nutrient_path.exists():
            logger.error(f"Food nutrient file not found: {food_nutrient_path}")
            return pd.DataFrame()
            
        # 读取并过滤food_nutrient.csv（分块读取大文件）
        logger.info(f"Reading food nutrient data from {food_nutrient_path}")
        chunks = []
        for chunk in pd.read_csv(food_nutrient_path, 
                                usecols=['fdc_id', 'nutrient_id', 'amount'],
                                chunksize=100000):
            # 过滤target_nutrients
            filtered_chunk = chunk[chunk['nutrient_id'].isin(target_nutrients)]
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
        
        # 合并所有chunks
        if chunks:
            food_nutrient_df = pd.concat(chunks, ignore_index=True)
        else:
            logger.warning("No food nutrient data found")
            return pd.DataFrame()
        
        # 创建完整索引
        complete_index = pd.MultiIndex.from_product(
            [food_df['fdc_id'].unique(), target_nutrients],
            names=['fdc_id', 'nutrient_id']
        )
        
        # 设置索引并用0填充缺失值
        food_nutrient_df = (food_nutrient_df
                           .set_index(['fdc_id', 'nutrient_id'])
                           .reindex(complete_index, fill_value=0)
                           .reset_index())
        
        # 读取nutrient.csv
        nutrient_path = DATA_DIR / "nutrient.csv"
        if not nutrient_path.exists():
            logger.error(f"Nutrient file not found: {nutrient_path}")
            return pd.DataFrame()
            
        # 读取并过滤nutrient.csv
        logger.info(f"Reading nutrient data from {nutrient_path}")
        nutrient_df = pd.read_csv(nutrient_path, usecols=['id', 'name', 'unit_name'])
        nutrient_df = nutrient_df[nutrient_df['id'].isin(target_nutrients)]
        nutrient_df = nutrient_df.rename(columns={'id': 'nutrient_id'})
        
        # 合并数据
        logger.info("Merging food, nutrient, and food_nutrient data")
        merged_df = (food_df
                    .merge(food_nutrient_df, on='fdc_id')
                    .merge(nutrient_df, on='nutrient_id'))
        
        # 调整列顺序
        merged_df = merged_df[['fdc_id', 'description', 'food_category_id', 'nutrient_id', 'name', 'amount', 'unit_name']]
        
        # 保存结果
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to {output_path}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging food data: {str(e)}", exc_info=True)
        return pd.DataFrame()


def add_food_category_labels(protein_target: List[int], 
                           subprotein_target: List[int], 
                           vegetable_target: List[int], 
                           fruits_target: List[int], 
                           fat_target: List[int], 
                           carb_target: List[int], 
                           supplement_target: List[int],
                           output_path: Optional[str] = None) -> pd.DataFrame:
    """
    添加食物类别标签，生成food_selected.csv文件
    
    参数:
    protein_target: List[int] - 蛋白质类食物ID列表
    subprotein_target: List[int] - 辅助蛋白类食物ID列表
    vegetable_target: List[int] - 蔬菜类食物ID列表
    fruits_target: List[int] - 水果类食物ID列表
    fat_target: List[int] - 脂肪类食物ID列表
    carb_target: List[int] - 碳水类食物ID列表
    supplement_target: List[int] - 补充剂类食物ID列表
    output_path: str - 输出文件路径，如果为None则使用默认路径
    
    返回:
    pd.DataFrame - 添加类别标签后的数据框
    """
    try:
        # 确定输出路径
        if output_path is None:
            output_path = DATA_DIR / "food_selected.csv"
        else:
            output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(exist_ok=True)
        
        # 合并所有目标食物ID
        all_target_ids = (protein_target + subprotein_target + vegetable_target + 
                          fruits_target + fat_target + carb_target + supplement_target)
        
        # 读取food_data.csv
        food_data_path = DATA_DIR / "food_data.csv"
        if not food_data_path.exists():
            logger.error(f"Food data file not found: {food_data_path}")
            return pd.DataFrame()
        
        # 读取并过滤food_data.csv
        logger.info(f"Reading food data from {food_data_path}")
        food_data_df = pd.read_csv(food_data_path)
        
        # 只保留目标食物ID
        food_data_df = food_data_df[food_data_df['fdc_id'].isin(all_target_ids)]
        
        # 创建一个新的列用于存储食物分类标签
        food_data_df['food_category_label'] = 'other'
        
        # 根据fdc_id列表添加分类标签
        food_data_df.loc[food_data_df['fdc_id'].isin(protein_target), 'food_category_label'] = 'protein'
        food_data_df.loc[food_data_df['fdc_id'].isin(subprotein_target), 'food_category_label'] = 'subprotein'
        food_data_df.loc[food_data_df['fdc_id'].isin(vegetable_target), 'food_category_label'] = 'vegetable'
        food_data_df.loc[food_data_df['fdc_id'].isin(fruits_target), 'food_category_label'] = 'fruit'
        food_data_df.loc[food_data_df['fdc_id'].isin(fat_target), 'food_category_label'] = 'fat'
        food_data_df.loc[food_data_df['fdc_id'].isin(carb_target), 'food_category_label'] = 'carbohydrate'
        food_data_df.loc[food_data_df['fdc_id'].isin(supplement_target), 'food_category_label'] = 'supplement'
        
        # 去重，保留每个fdc_id的第一条记录
        food_data_df = food_data_df.drop_duplicates(subset=['fdc_id'])
        
        # 保存结果
        food_data_df.to_csv(output_path, index=False)
        logger.info(f"Food category labels added and saved to {output_path}")
        
        return food_data_df
        
    except Exception as e:
        logger.error(f"Error adding food category labels: {str(e)}", exc_info=True)
        return pd.DataFrame()


def export_food_data_stats(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    导出食物数据统计信息
    
    参数:
    output_path: str - 输出文件路径，如果为None则使用默认路径
    
    返回:
    Dict[str, Any] - 统计信息字典
    """
    try:
        # 确定输出路径
        if output_path is None:
            output_path = DATA_DIR / "food_data_stats.json"
        else:
            output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(exist_ok=True)
        
        # 读取food_data.csv
        food_data_path = DATA_DIR / "food_data.csv"
        if not food_data_path.exists():
            logger.error(f"Food data file not found: {food_data_path}")
            return {}
        
        # 读取food_data.csv
        food_data_df = pd.read_csv(food_data_path)
        
        # 读取food_selected.csv
        food_selected_path = DATA_DIR / "food_selected.csv"
        if not food_selected_path.exists():
            logger.warning(f"Food selected file not found: {food_selected_path}")
            food_selected_df = pd.DataFrame()
        else:
            food_selected_df = pd.read_csv(food_selected_path)
        
        # 计算统计信息
        stats = {
            'total_foods': len(food_data_df['fdc_id'].unique()),
            'total_nutrients': len(food_data_df['nutrient_id'].unique()),
            'categorized_foods': len(food_selected_df) if not food_selected_df.empty else 0,
            'category_distribution': {},
            'nutrient_stats': {},
            'generated_date': datetime.now().isoformat()
        }
        
        # 计算类别分布
        if not food_selected_df.empty and 'food_category_label' in food_selected_df.columns:
            category_counts = food_selected_df['food_category_label'].value_counts().to_dict()
            stats['category_distribution'] = category_counts
        
        # 计算营养素统计
        for nutrient_id in food_data_df['nutrient_id'].unique():
            nutrient_df = food_data_df[food_data_df['nutrient_id'] == nutrient_id]
            
            if not nutrient_df.empty:
                nutrient_name = nutrient_df['name'].iloc[0]
                nutrient_unit = nutrient_df['unit_name'].iloc[0]
                
                # 计算营养素的基本统计
                amount_stats = nutrient_df['amount'].describe().to_dict()
                
                stats['nutrient_stats'][str(nutrient_id)] = {
                    'name': nutrient_name,
                    'unit': nutrient_unit,
                    'min': amount_stats['min'],
                    'max': amount_stats['max'],
                    'mean': amount_stats['mean'],
                    'std': amount_stats['std'],
                    'count': amount_stats['count']
                }
        
        # 保存结果
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Food data statistics exported to {output_path}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error exporting food data statistics: {str(e)}", exc_info=True)
        return {}




def merge_food_data():

    description_contains = ['canned', 'fried', 'with salt', 'juice', 'smoked', 
                            'dried', 'forzen', 'uncooked', 'seeds']
    pattern = '|'.join(description_contains)
    # 读取food.csv并保留指定列
    food_df = (pd.read_csv('./data/food.csv', usecols=['fdc_id', 'description', 'food_category_id'])
               .query('food_category_id  in @target_food_category')
               # 新增raw数据过滤
               .pipe(lambda df: df[~df['description'].str.contains(pattern, case=False, na=False)]))

    # 读取food_nutrient.csv并保留指定列
    target_nutrients = [1003, 1004, 1005, 1008, 1087, 1088, 1089, 1090, 1091, 
                        1092, 1093, 1095, 1098, 1100, 1101, 1103, 1104, 1110, 
                        1109, 1165, 1166, 1167, 1170, 1171, 1178, 1180, 1186, 
                        1210, 1211, 1212, 1213, 1214, 1215, 1217, 1219, 1220, 
                        1221, 1229, 1230, 1272, 1278, 1316, 1404]
    
    food_nutrient_df = (pd.read_csv('./data/food_nutrient.csv', usecols=['fdc_id', 'nutrient_id', 'amount'])
                        .pipe(lambda df: df[df['nutrient_id'].isin(target_nutrients)])
                        .set_index(['fdc_id', 'nutrient_id'])
                        .reindex(pd.MultiIndex.from_product([food_df['fdc_id'].unique(), target_nutrients], names=['fdc_id', 'nutrient_id']), fill_value=0).reset_index())
    
    # 读取nutrient.csv并保留指定列
    nutrient_df = (pd.read_csv('./data/nutrient.csv', usecols=['id', 'name', 'unit_name'])
                   .query('id in @target_nutrients')
                   .rename(columns={'id': 'nutrient_id'}))
    
    print(nutrient_df.head())

    # 预定义的有毒食物列表
    toxic_foods = [ "chocolate", "coffee", "grape", "raisin", "onion", "garlic", "xylitol",
                    "alcohol", "avocado", "caffeine", "mushroom", "macadamia", "tea", "cocoa powder",
                    "cocoa beans", "cherries", "cherries pits", "cherry", "citrus", "lemons",
                    "limes", "oranges", "apple seed", "apple core", "fig", "onion", "leeks",
                    "chives", "scallions", "shallots", "macadamia nut", "walnuts", "almonds",
                    "pistchios", "cashews", "peanuts", "cooked bone", "milk", "cheese",
                    "dairy", "bacon", "sausage", "duck skin", "hot dogs", "deil meat",
                    "raw pottatoes", "green tomatoes", "eggplant", "mushroom", "Aubergine",
                    "Raisins"]
    
    print("Food nutrient columns:", food_nutrient_df.columns.tolist())
    print("Nutrient df columns:", nutrient_df.columns.tolist())

    # 合并数据
    merged_df = (
        food_df
            .merge(food_nutrient_df, on='fdc_id')
            .merge(nutrient_df, on='nutrient_id')
            .assign(
                is_hazardous=lambda x:x['description'].str.lower().str.contains('|'.join(toxic_foods))
                    )
    )

    # 调整保存列顺序
    merged_df = merged_df[['fdc_id', 'description', 'food_category_id', 'nutrient_id', 'name', 'amount', 'unit_name']]
    
    # 创建输出目录
    import os
    os.makedirs('data', exist_ok=True)
    
    # 保存合并结果
    merged_df.to_csv('./data/food_data.csv', index=False)
    return merged_df

def add_food_category_labels(protein_target, subprotein_target, vegetable_target, fruits_target, fat_target, carb_target, supplement_target):
    """
    将食物ID列表与food_data.csv中的数据匹配，并添加食物分类标签
    
    参数:
    protein_target: 蛋白质类食物ID列表
    vegetable_target: 蔬菜类食物ID列表
    fruits_target: 水果类食物ID列表
    fat_target: 脂肪类食物ID列表
    carb_target: 碳水化合物类食物ID列表
    
    返回:
    带有食物分类标签的DataFrame
    """
    # 合并所有目标食物ID
    all_target_ids = protein_target + vegetable_target + fruits_target + fat_target + carb_target + subprotein_target + supplement_target
    
    # 读取food_data.csv，只保留目标食物ID
    food_data_df = pd.read_csv('./data/food_data.csv')
    food_data_df = food_data_df[food_data_df['fdc_id'].isin(all_target_ids)]
    
    # 读取food.csv，获取food_category_id
    food_df = pd.read_csv('./data/food.csv', usecols=['fdc_id', 'food_category_id'])
    
    # 将food_category_id合并到food_data_df中，使用suffixes参数避免列名重复
    food_data_df = food_data_df.merge(food_df, on='fdc_id', how='left', suffixes=('', '_from_food'))
    
    # 创建一个新的列用于存储食物分类标签
    food_data_df['food_category_label'] = 'other'
    
    # 根据fdc_id列表添加分类标签
    food_data_df.loc[food_data_df['fdc_id'].isin(protein_target), 'food_category_label'] = 'protein'
    food_data_df.loc[food_data_df['fdc_id'].isin(vegetable_target), 'food_category_label'] = 'vegetable'
    food_data_df.loc[food_data_df['fdc_id'].isin(fruits_target), 'food_category_label'] = 'fruit'
    food_data_df.loc[food_data_df['fdc_id'].isin(fat_target), 'food_category_label'] = 'fat'
    food_data_df.loc[food_data_df['fdc_id'].isin(carb_target), 'food_category_label'] = 'carbohydrate'
    food_data_df.loc[food_data_df['fdc_id'].isin(subprotein_target), 'food_category_label'] = 'subprotein'
    food_data_df.loc[food_data_df['fdc_id'].isin(supplement_target), 'food_category_label'] = 'supplement'
    
    # 保存结果
    food_data_df.to_csv('./data/food_selected.csv', index=False)
    
    return food_data_df

def AAFCO_constraints(state="adult"):
    adult_nutrient_req = {
        1003: {"min": 45.0,     "max": None},
        1004: {"min": 13.8,     "max": None},
        1008: {"min": 2000,     "max": None},
        1087: {"min": 1.25,     "max": 6.25},
        1089: {"min": 10,       "max": None},     # mg
        1090: {"min": 0.15,     "max": None},
        1091: {"min": 1.0,      "max": 4.0},
        1092: {"min": 1.5,      "max": None},
        1093: {"min": 0.2,      "max": None},
        1095: {"min": 20,       "max": None},     # mg
        1098: {"min": 1.83,     "max": None},   # mg
        1103: {"min": 0.0005,   "max": 0.5},
        1104: {"min": 1250,     "max": 62500},
        1110: {"min": 125,      "max": 750},
        1109: {"min": 12.5/1.49, "max": None},
        1165: {"min": 0.56,     "max": None},   # mg Vitamin-B1
        1178: {"min": 0.007,    "max": None},  # mg
        1166: {"min": 1.3,      "max": None},
        1316: {"min": 3.3,      "max": None},    # LA
        1404: {"min": 0.2,      "max": None}     # ALA
    }
    return adult_nutrient_req

def process_food_nutrients(food_df, target_nutrients, chunk_size=100000):
    # Initialize an empty list to store chunks
    chunks = []
    
    # Read and process the file in chunks
    for chunk in pd.read_csv('./data/food_nutrient.csv', 
                           usecols=['fdc_id', 'nutrient_id', 'amount'],
                           chunksize=chunk_size):
        # Filter for target nutrients
        filtered_chunk = chunk[chunk['nutrient_id'].isin(target_nutrients)]
        if not filtered_chunk.empty:
            chunks.append(filtered_chunk)
    
    # Combine all chunks
    if chunks:
        food_nutrient_df = pd.concat(chunks, ignore_index=True)
    else:
        # If no data found, create empty DataFrame with correct columns
        food_nutrient_df = pd.DataFrame(columns=['fdc_id', 'nutrient_id', 'amount'])
    
    # Create complete index of all combinations
    complete_index = pd.MultiIndex.from_product(
        [food_df['fdc_id'].unique(), target_nutrients],
        names=['fdc_id', 'nutrient_id']
    )
    
    # Set index and reindex with complete combinations
    food_nutrient_df = (food_nutrient_df
                       .set_index(['fdc_id', 'nutrient_id'])
                       .reindex(complete_index, fill_value=0)
                       .reset_index())
    
    return food_nutrient_df

def merge_specific_food_data(target_fdc_ids, target_nutrient_ids, output_filename='merged_food_data.csv'):
    """
    合并特定食物和营养素的数据，生成包含完整食物营养信息的CSV文件
    
    该函数从三个数据源（food.csv、food_nutrient.csv和nutrient.csv）中提取数据，
    并根据指定的食物ID和营养素ID进行过滤和合并，生成一个包含完整食物营养信息的CSV文件。
    
    参数:
    ----------
    target_fdc_ids : list
        需要包含的食物ID列表，这些ID对应food.csv中的fdc_id列
    target_nutrient_ids : list
        需要包含的营养素ID列表，这些ID对应nutrient.csv中的id列
    output_filename : str, optional
        输出文件的名称，默认为'merged_food_data.csv'
        
    返回:
    ----------
    pandas.DataFrame
        合并后的数据框，包含以下列：
        - fdc_id: 食物ID
        - description: 食物描述
        - food_category_id: 食物类别ID
        - nutrient_id: 营养素ID
        - name: 营养素名称
        - amount: 营养素含量
        - unit_name: 营养素单位
        
    示例:
    ----------
    >>> target_foods = [1001, 1002, 1003]
    >>> target_nutrients = [1003, 1004, 1005]
    >>> result_df = merge_specific_food_data(target_foods, target_nutrients, 'my_food_data.csv')
    >>> print(result_df.head())
    """
    # 1. 从 food.csv 获取指定 fdc_id 的信息
    food_df = (pd.read_csv('./data/food.csv', usecols=['fdc_id', 'description', 'food_category_id'])
               .query('fdc_id in @target_fdc_ids'))
    
    # 2. 从 food_nutrient.csv 获取营养素信息（使用chunks处理大文件）
    chunks = []
    for chunk in pd.read_csv('./data/food_nutrient.csv', 
                           usecols=['fdc_id', 'nutrient_id', 'amount'],
                           chunksize=100000):
        # 过滤出目标 fdc_ids 和 nutrient_ids
        filtered_chunk = chunk[
            (chunk['fdc_id'].isin(target_fdc_ids)) & 
            (chunk['nutrient_id'].isin(target_nutrient_ids))
        ]
        if not filtered_chunk.empty:
            chunks.append(filtered_chunk)
    
    # 合并所有chunks
    if chunks:
        food_nutrient_df = pd.concat(chunks, ignore_index=True)
    else:
        food_nutrient_df = pd.DataFrame(columns=['fdc_id', 'nutrient_id', 'amount'])
    
    # 创建完整的组合索引并填充缺失值为0
    complete_index = pd.MultiIndex.from_product(
        [target_fdc_ids, target_nutrient_ids],
        names=['fdc_id', 'nutrient_id']
    )
    food_nutrient_df = (food_nutrient_df
                       .set_index(['fdc_id', 'nutrient_id'])
                       .reindex(complete_index, fill_value=0)
                       .reset_index())
    
    # 3. 从 nutrient.csv 获取营养素名称和单位信息
    nutrient_df = (pd.read_csv('./data/nutrient.csv', usecols=['id', 'name', 'unit_name'])
                   .query('id in @target_nutrient_ids')
                   .rename(columns={'id': 'nutrient_id'}))
    
    # 4. 合并所有数据
    merged_df = (food_df
                .merge(food_nutrient_df, on='fdc_id')
                .merge(nutrient_df, on='nutrient_id'))
    
    # 调整列顺序
    merged_df = merged_df[['fdc_id', 'description', 'food_category_id', 
                          'nutrient_id', 'name', 'amount', 'unit_name']]
    
    # 保存结果
    merged_df.to_csv(output_filename, index=False)
    return merged_df

# 预编译正则表达式模式
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
        # 创建更精确的匹配模式
        pattern = r'\b' + re.escape(food.lower()) + r'\b'
        patterns[food] = re.compile(pattern)
    return patterns

@lru_cache(maxsize=128)
def load_food_data() -> pd.DataFrame:
    """
    加载食物数据，使用缓存避免重复读取
    
    返回:
    pd.DataFrame - 食物数据
    """
    try:
        return pd.read_csv('./data/food_selected.csv', usecols=['fdc_id', 'description'])
    except Exception as e:
        logger.error(f"Error loading food data: {str(e)}")
        raise

def find_allergic_foods(allergy_foods: List[str], min_confidence: float = 0.8) -> Dict[str, List[int]]:
    """
    在food.csv中查找包含过敏食物关键词的食物ID，返回分类结果
    
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
        
        # 加载食物数据
        food_df = load_food_data()
        
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
            logger.info(f"{food}: {len(ids)} matches")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in find_allergic_foods: {str(e)}")
        raise

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

if __name__ == "__main__":
    # result_df = merge_food_data()
    # print("数据合并完成，结果已保存到 data/food_data.csv")
    # print("合并后数据样例：")
    # print(result_df.head())
    # constraints = AAFCO_constraints()

    # for nut, bounds in constraints.items():
    #     print("nut. {nut}", nut)
    #     print("bounds. {bounds}", bounds)
    #     print("anoumt. {bounds[nut]}", bounds['min'])
    #     # print(food_dict[food]['nutrients'][] for food in food_dict)

    # # 示例使用
    # target_fdc_ids = []  # 添加你的目标 fdc_ids
    # target_nutrient_ids = []  # 添加你的目标 nutrient_ids
    # result_df = merge_specific_food_data(target_fdc_ids, target_nutrient_ids)
    # print("数据合并完成！")
    # print("\n合并后数据样例：")
    # print(result_df.head())

    target_fcd_id = protein_target + subprotein_target + vegetable_target + fruits_target + fat_target + carb_target + supplement_target
    # 合并特定食物数据
    # print("开始合并数据...")
    # result_df = merge_specific_food_data(target_fcd_id, target_nutrients, output_filename='data/food_data.csv')
    # print("数据合并完成！")
    # print("\n合并后数据样例：")
    # print(result_df.head())
    # foods_len = len(fruits_target) + len(protein_target) + len(vegetable_target) + len(fat_target) + len(carb_target) + len(subprotein_target)
    # print('total type of foods is', foods_len)

    print('开始添加食物分类标签...')
    add_food_category_labels(protein_target, subprotein_target, vegetable_target, fruits_target, fat_target, carb_target, supplement_target)
    print('食物分类标签添加完成！')

    # # 获取所有过敏食物的fdc_id集合
    # find_id = get_all_allergic_fdc_ids(['beef', 'pork', "chocolate", "coffee", "grape", "raisin", "onion", "garlic", "xylitol",
    #                 "alcohol", "avocado", "caffeine", "mushroom", "macadamia", "tea", "cocoa powder",
    #                 "cocoa beans", "cherries", "cherries pits", "cherry", "citrus", "lemons",
    #                 "limes", "oranges", "apple seed", "apple core", "fig", "onion", "leeks",
    #                 "chives", "scallions", "shallots", "macadamia nut", "walnuts", "almonds",
    #                 "pistchios", "cashews", "peanuts", "cooked bone", "milk", "cheese",
    #                 "dairy", "bacon", "sausage", "duck skin", "hot dogs", "deil meat",
    #                 "raw pottatoes", "green tomatoes", "eggplant", "mushroom", "Aubergine",
    #                 "Raisins"])
    
    # print('find allergy food id :', find_id)