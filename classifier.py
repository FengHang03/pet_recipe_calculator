import pandas as pd
from collections import defaultdict

def classify_foods(df):
    """食物分类函数"""
    # 分类1：按原始食品类别
    df['category'] = df['food_category_id'].astype('category')
    
    # 分类2：按蛋白质含量分级
    protein_mask = df['nutrient_id'] == 1003
    df['protein_level'] = pd.cut(df[protein_mask]['amount'],
                               bins=[0, 10, 20, 100],
                               labels=['低蛋白', '中蛋白', '高蛋白'],
                               right=False)
    
    # 分类3：按脂肪含量分级
    fat_mask = df['nutrient_id'] == 1004
    df['fat_level'] = pd.cut(df[fat_mask]['amount'],
                           bins=[0, 5, 20, 100],
                           labels=['低脂', '中脂', '高脂'],
                           right=False)
    
    # 分类4：按食物形态（基于description）
    form_categories = {
        '肉': ['chicken', 'beef', 'pork', '肉'],
        '鱼': ['fish', 'salmon', 'tuna', '鱼'],
        '谷物': ['rice', 'oat', 'wheat', '谷物'],
        '蔬菜': ['vegetable', 'carrot', 'spinach', '蔬菜'],
        '乳制品': ['milk', 'cheese', 'yogurt', '乳']
    }
    
    df['food_form'] = '其他'
    for form, keywords in form_categories.items():
        mask = df['description'].str.lower().str.contains('|'.join(keywords), na=False)
        df.loc[mask, 'food_form'] = form
    
    # 分类5：营养密度评分
    nutrient_weights = {
        1003: 0.3,  # 蛋白质
        1004: -0.2, # 脂肪（负权重）
        1087: 0.2,  # 钙
        1089: 0.2,  # 铁
        1008: 0.1   # 能量
    }
    
    score_df = df.pivot(index='fdc_id', columns='nutrient_id', values='amount')
    df['nutrition_score'] = score_df.apply(
        lambda x: sum(x.get(nid, 0)*weight for nid, weight in nutrient_weights.items()),
        axis=1
    )
    
    # 新增分类6：功能性分类
    df = classify_functional(df)
    
    return df

def classify_functional(df):
    """功能性分类"""
    # 营养成分与功能映射
    functional_nutrients = {
        '消化健康': [
            (1079, 3),   # 膳食纤维 ≥3g
            (1005, 5),   # 碳水化合物 ≥5g
            (1185, 1)    # 益生菌 ≥1M CFU/g
        ],
        '皮毛健康': [
            (1103, 0.5),  # Omega-3 ≥0.5g 
            (1104, 0.3),  # Omega-6 ≥0.3g
            (1162, 0.1)   # 维生素E ≥0.1mg
        ],
        '骨骼关节': [
            (1087, 100),  # 钙 ≥100mg
            (1088, 80),    # 磷 ≥80mg
            (1175, 0.5)    # 葡萄糖胺 ≥0.5g
        ],
        '免疫支持': [
            (1165, 10),    # 维生素C ≥10mg
            (1159, 5),     # 锌 ≥5mg
            (1114, 50)     # β-胡萝卜素 ≥50μg
        ],
        '心脏健康': [
            (1124, 0.1),   # 牛磺酸 ≥0.1g
            (1092, 300),   # 钾 ≥300mg
            (1004, 5)      # 脂肪 ≤5g (上限)
        ]
    }
    
    # 创建营养含量数据透视表
    pivot_df = df.pivot(index='fdc_id', columns='nutrient_id', values='amount')
    
    # 初始化分类列
    df['functional_class'] = ''
    
    # 遍历每个功能类别
    for func, conditions in functional_nutrients.items():
        mask = pd.Series(True, index=pivot_df.index)
        
        for nutrient_id, threshold in conditions:
            if nutrient_id in pivot_df.columns:
                # 处理上限条件（脂肪特殊处理）
                if nutrient_id == 1004 and func == '心脏健康':
                    mask &= (pivot_df[nutrient_id] <= threshold)
                else:
                    mask &= (pivot_df[nutrient_id] >= threshold)
            else:
                print(f"警告：缺少营养成分ID {nutrient_id}，可能影响{func}分类准确性")
        
        # 更新符合条件的记录
        func_ids = pivot_df[mask].index
        df.loc[df['fdc_id'].isin(func_ids), 'functional_class'] += func + ';'
    
    # 清理分类字符串
    df['functional_class'] = df['functional_class'].str.rstrip(';')
    
    return df

# 使用示例
if __name__ == "__main__":
    df = pd.read_csv('./data/food_data.csv')
    classified_df = classify_foods(df)
    
    # 保存分类结果
    classified_df.to_csv('./data/classified_food_data.csv', index=False)
    print("分类完成，新增5个分类维度：")
    print(classified_df[['fdc_id', 'description', 'category', 'protein_level', 
                       'fat_level', 'food_form', 'nutrition_score']].head()) 