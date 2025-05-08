import pandas as pd
import math

class Pet:
    def __init__(self, breed, age, weight, gender, allergy='', ideal_weight = 0, weight_unit = 'lb', age_unit = 'month', life_stage=0, activity_level=1, pup_number=5, lactation_weeks=1):
        self.breed = breed  # 宠物品类 0 for "Toy", 1 for "Small", 2 for "Medium", 3 for "Large"
        self.life_stage = life_stage    # 宠物生命阶段 0 for "normal", 1 for "neuter", 2 for "pregnancy", 3 for "lactation"
        self.weight = int(weight)  # 宠物体重（kg）
        self.gender = gender    # 宠物性别 0 for "male", 1 for "female"
        self.activity_level = activity_level  # 活动水平: ['activity_level_sedentary', 'activity_level_light', 'activity_level_moderatel', 'activity_level_highly', 'activity_level_extremely']
        self.ideal_weight = ideal_weight  # 宠物理想体重
        self.pup_number = pup_number
        self.lactation_weeks = lactation_weeks
        self.weight_unit = weight_unit
        self.allergy = allergy,  # 过敏食物
        self.energy = 0,    # 宠物所需能量（计算值）

        if age_unit == 'month':
            self.age_months = int(age)
        elif age_unit == 'year':
            self.age_months = int(age) * 12
    def age_years(self):
        """
        Return the age of the pet in years.
        """
        return self.age_months / 12

    def weight_kg(self):
        if self.weight_unit == 'lb':
            return self.weight * 0.45359231
        else:
            return self.weight
    def is_puppy(self):
        return self.age_months < 12
    
    def is_adult(self):
        # if self.breed == 'normal':
        #     return self.age_months > 12 * 7
        # else:
        #     return self.age_months > 12 * 6
        return self.age_months < 12 * 7
    
    def calculate_activity_level(self):
        if self.activity_level == 'activity_level_sedentary':
            return 1.4
        elif self.activity_level == 'activity_level_light':
            return 1.6
        elif self.activity_level == 'activity_level_moderatel':
            return 1.8
        elif self.activity_level == 'activity_level_highly':
            return 2.0
        elif self.activity_level == 'activity_level_extremely':
            return 2.2
        else:
            return 1.8

    def calculate_lactation_factor(self):
    
        if self.lactation_weeks == 1:
            self.lactation_factor = 0.75 # 哺乳一周
        elif self.lactation_weeks == 2:
            self.lactation_factor = 0.95 # 哺乳二周
        elif self.lactation_weeks == 3:
            self.lactation_factor = 1.1 # 哺乳三周
        elif self.lactation_weeks == 4:
            self.lactation_factor = 1.2 # 哺乳四周
        
        return self.lactation_factor

    def calculate_bmr(self):
        """
        计算基础代谢率 (BMR)，可以根据宠物品类调整公式。
        """
        calculate_weight = self.weight_kg()
        bmr = 70 * calculate_weight ** 0.75

        return bmr

    def calculate_caloric_needs(self):
        """
        根据年龄、生命阶段、活动水平调整卡路里需求。
        """
        calculate_weight = self.weight_kg()
        bmr = self.calculate_bmr()

        # *************
        # 绝育 || 妊娠期 || 哺乳期 || 年龄 - 幼犬 || 年龄 - 老犬
        # *************
        # 绝育的犬能量需要
        if self.life_stage == 'life_stage_neuter':
            caloric_needs = 1.6 * bmr

        # 妊娠期犬能量需要
        if self.life_stage == 'life_stage_pregnancy':
            caloric_needs = 1.8 * bmr + 26 * calculate_weight

        # 哺乳期犬(断奶后4-14周)能量需要
        elif self.life_stage == 'life_stage_lactation':
            if self.pup_number <= 4:
                caloric_needs = 2 * bmr + calculate_weight * (24 * self.pup_number) * self.calculate_lactation_factor()
            else:
                caloric_needs = 2 * bmr + calculate_weight * (96 + 12 * (self.pup_number - 4)) * self.calculate_lactation_factor()
            
        # 年龄划分
        else:
            if self.is_puppy():
                # 幼犬断奶后每日代谢能需要量
                # caloric_needs = 1.8 * bmr * 3.2 * (math.e ** (-0.87 * (calculate_weight / self.ideal_weight)) - 0.1)
                if self.age_months <= 4:
                    caloric_needs = 3 * bmr
                else:
                    caloric_needs = 2.5 * bmr
            elif self.is_adult():
                # 成年犬
                
                caloric_needs = self.calculate_activity_level() * bmr
            else:
                # 老年犬
                caloric_needs = (self.calculate_activity_level() - 0.2) * bmr

        return caloric_needs

    def calculate_protein_rate(self):
        """
        计算蛋白质需求。
        一般成年犬的蛋白质需求为体重的 10%。
        """
        # 小型犬与中小型犬，判断其年龄阶段所需蛋白质比例
        if self.is_puppy():
            protein_rate = [0.225, 0.40]
        elif self.is_adult():
            protein_rate = [0.18, 0.40]
        else:
            protein_rate = [0.20, 0.40]

        # 大型犬，判断其年龄阶段所需蛋白质比例
        # if self.is_puppy():
        #     protein_rate = [0.22, 0.30]
        # elif self.is_adult():
        #     protein_rate = [0.18, 0.30]
        # else:
        #     protein_rate = [0.20, 0.30]

        return protein_rate

    def calculate_fat_rate(self):
        """
        计算脂肪需求。
        一般成年犬的脂肪需求为体重的 20%。
        """
        caloric_needs = self.calculate_caloric_needs()

        # 小型犬与中小型犬，判断其年龄阶段所需蛋白质比例
        if self.is_puppy():
            fat_rate = [0.225, 0.30]
        elif self.is_adult():
            fat_rate = [0.18, 0.30]
        else:
            fat_rate = [0.20, 0.30]

        # 大型犬，判断其年龄阶段所需蛋白质比例
        # if self.is_puppy():
        #     fat_rate = [0.085, 0.25]
        # elif self.is_adult():
        #     fat_rate = [0.055, 0.25]
        # else:
        #     fat_rate = [0.055, 0.25]

        return fat_rate

if __name__ == '__main__':
    # print("Hello Python!")
    # Pet1 = Pet("Alaskan Klee Kai(Toy)", 18, 10, 0, 35, life_stage = 0)
    pet_info = {
        'activity_level' : "normal",
        'age': "30",
        'breed' : "博美犬",
        'gender' : "female",
        'life_stage' : "normal",
        'weight' : "11" 
    }
    pet_info_list = {
            'breed': '未知品种',
            'age': 15,
            'weight': 10,
            'gender': 'male',
            'life_stage': 'normal',
            'activity_level': 'normal',
            'pup_number': 4,
            'lactation_weeks': 1
        }
    energy_level = {
        'Not lively': 0, 
        "less lively": 1, 
        "normal": 2,
        "lively": 3,
        "very lively": 4
    }
    for key, value in pet_info_list.items():
        if key not in pet_info:
            pet_info[key] = value
        if key == 'activity_level':
            pet_info[key] = energy_level[value]
    print(pet_info)
    my_pet = Pet(breed = pet_info['breed'], age=pet_info['age'], weight=pet_info['weight'], gender=pet_info['gender'], life_stage=pet_info['life_stage'], activity_level=pet_info['activity_level'], pup_number=pet_info['pup_number'], lactation_weeks=pet_info['lactation_weeks'])
    print("BMR:", my_pet.calculate_bmr())
    print("Caloric Needs:", my_pet.calculate_caloric_needs())
    print("Protein Rate:", my_pet.calculate_protein_rate())
    print("Fat Rate:", my_pet.calculate_fat_rate())
    print(int(pet_info['weight']))
    
#     print('recipe[0]: ', recipe[0])
#     # for sol in recipe:
#     #         print(f"Solution #{sol['solution_index']}")
#     #         print(f"Total Calories: {sol['total_calories']:.0f} kcal")
#     #         # 显示每种食材及其数量
#     #         print("Recomended Foods:")
#     #         print("type of chosen_foods: ", type(sol["chosen_foods"]))
#     #         for f in sol["chosen_foods"]:
#     #             print(type(f))
#     #             print(f)
#     #             for key, value in f.items():
#     #                 print(f"{key}: {value} g") 

#     # for r in recipe:
#     #     foods = r['chosen_foods']
#     #     print("type of foods[1]: ", type(foods[1]))
#     #     print(foods)
#     #     for f in foods:
#     #         print(type(f))
#     #         print(f)
#     #         # 显示f的键和值
#     #         for key, value in f.items():
#     #             print(key, value)
