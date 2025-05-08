import pandas as pd
import os
from data_operate import add_food_category_labels

def generate_nutrient_list():
    # 读取数据
    target_nutrients = [1003, 1004, 1005, 1008, 1087, 1088, 1089, 1090, 1091, 
                        1092, 1093, 1095, 1098, 1100, 1101, 1103, 1104, 1110, 
                        1109, 1165, 1166, 1167, 1170, 1171, 1178, 1180, 1186, 
                        1210, 1211, 1212, 1213, 1214, 1215, 1217, 1219, 1220, 
                        1221, 1229, 1230, 1272, 1278, 1316, 1404]
    
    nutrient_df = (pd.read_csv('data/nutrient.csv', usecols=['id', 'name', 'unit_name'])
                   .query('id in @target_nutrients')
                   .rename(columns={'id': 'nutrient_id'}))
    
    os.makedirs('data', exist_ok=True)
    nutrient_df.to_csv('data/nutrient_list_AAFCO.csv', index=False)
    return nutrient_df

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


if __name__ == "__main__":
    # 添加食物分类标签
    # food_df_with_categories = add_food_category_labels(
    #     protein_target, vegetable_target, fruits_target, fat_target, carb_target, subprotein_target
    # )
    # print("食物分类标签已添加并保存到 data/food_with_categories.csv")
    # print("\n食物分类统计:")
    # print(food_df_with_categories['food_category_label'].value_counts())
    
    # # 生成各种食物列表
    # result_df = merge_specific_food_data(fruits_target, target_nutrients, output_filename='data/fruits_food.csv')
    # merge_specific_food_data(protein_target, target_nutrients, output_filename='data/protein_food.csv')
    # merge_specific_food_data(vegetable_target, target_nutrients, output_filename='data/vegetable_food.csv')
    # merge_specific_food_data(fat_target, target_nutrients, output_filename='data/fat_food.csv')
    # merge_specific_food_data(carb_target, target_nutrients, output_filename='data/carb_food.csv')
    # merge_specific_food_data(subprotein_target, target_nutrients, output_filename='data/subprotein_food.csv')