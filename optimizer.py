from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import numpy as np
import pandas as pd

def optimize_recipe(food_dict, constraints):
    """
    线性规划优化函数
    :param food_dict: 食物字典 {fdc_id: {nutrient_id: amount, ...}, ...}
    :param constraints: 约束条件 {nutrient_id: {'min': float, 'max': float}, ...}
    :return: 优化结果 {fdc_id: amount, ...}
    """
    # 初始化线性规划问题
    prob = LpProblem("PetFoodOptimization", LpMinimize)
    
    # 创建决策变量（食物用量，非负）
    food_vars = LpVariable.dicts("Food", food_dict.keys(), lowBound=0, cat='Continuous')
    
    # 设置目标函数（最小化总食物量）
    prob += lpSum([food_vars[fid] for fid in food_dict.keys()])
    
    # 添加营养约束
    for nutrient_id, bounds in constraints.items():
        if 'min' in bounds:
            prob += lpSum(food_vars[fid] * food_dict[fid].get(nutrient_id, 0) 
                        for fid in food_dict.keys()) >= bounds['min']
        if 'max' in bounds:
            prob += lpSum(food_vars[fid] * food_dict[fid].get(nutrient_id, 0) 
                        for fid in food_dict.keys()) <= bounds['max']
    
    # 求解问题
    prob.solve()
    
    # 提取结果
    if prob.status == 1:  # 1表示最优解
        return {fid: var.varValue for fid, var in food_vars.items() if var.varValue > 1e-5}
    else:
        raise ValueError("No feasible solution found")

def load_data():
    """
    加载并预处理数据
    :return: (food_dict, nutrient_info)
    """
    # 读取合并后的数据
    df = pd.read_csv('./data/food_data.csv')
    
    # 构建食物字典
    food_dict = {}
    for (fdc_id, description), group in df.groupby(['fdc_id', 'description']):
        food_dict[fdc_id] = {
            'description': description,
            'nutrients': group.set_index('nutrient_id')['amount'].to_dict()
        }
    
    # 获取营养成分元数据
    nutrient_info = df[['nutrient_id', 'name', 'unit_name']].drop_duplicates().set_index('nutrient_id')
    
    return food_dict, nutrient_info.to_dict('index')

# 使用示例
if __name__ == "__main__":
    # 加载数据
    food_dict, nutrient_info = load_data()
    
    # 设置约束条件示例（需根据实际需求调整）
    constraints = {
        1003: {'min': 60},    # 蛋白质至少60g
        1004: {'max': 20},    # 脂肪不超过20g
        1008: {'min': 2000},  # 能量至少2000kcal
        1087: {'min': 800},   # 钙至少800mg
        1089: {'min': 10}     # 铁至少10mg
    }
    
    try:
        # 执行优化
        optimized = optimize_recipe(
            food_dict={fid: data['nutrients'] for fid, data in food_dict.items()},
            constraints=constraints
        )
        
        # 打印结果
        print("优化后的食谱组成：")
        total = sum(optimized.values())
        for fid, amount in optimized.items():
            print(f"{food_dict[fid]['description']}: {amount:.2f}g ({amount/total:.1%})")
            
    except ValueError as e:
        print(str(e)) 