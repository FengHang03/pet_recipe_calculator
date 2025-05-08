# 宠物食谱优化工具

一个基于线性规划的宠物食谱优化系统，可根据营养需求、过敏限制和食物分类生成平衡的宠物食谱。

## 特性

- **营养平衡优化**: 基于AAFCO标准，确保宠物获得所需营养



 
- **过敏检测**: 自动识别和排除过敏食物
- **食物分类约束**: 保证食谱中有适量的蛋白质、碳水、脂肪、蔬果等
- **可配置参数**: 灵活调整优化权重和约束条件
- **结果可视化**: 直观展示食谱组成和营养满足度
- **命令行界面**: 方便的命令行工具，支持批处理
- **结果导出**: 支持多种格式导出优化结果

## 安装

1. 克隆代码库：
```bash
git clone https://github.com/yourusername/pet-recipe-optimizer.git
cd pet-recipe-optimizer
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据：
```bash
python -m data_utils merge_food_data
```

## 使用方法

### 命令行工具

最简单的使用方式是通过命令行：

```bash
python recipe_optimize.py --energy 1500 --stage adult --allergy pork chocolate
```

可用参数:
- `--energy`: 目标能量(kcal)，必需
- `--min-variety`: 最少食物种类，默认4
- `--max-variety`: 最多食物种类，默认8
- `--allergy`: 过敏食物关键词列表
- `--stage`: 宠物生命阶段，adult或child，默认adult
- `--alpha`: 能量偏差权重，默认100.0
- `--beta`: 营养素松弛变量权重，默认10.0
- `--variety-factor`: 食物多样性权重，默认1.0
- `--output`: 输出目录，默认output
- `--plot`: 绘制结果图表
- `--export`: 导出结果到文件
- `--log-level`: 日志级别，默认INFO

### 作为模块导入

您也可以在自己的Python代码中导入并使用该模块：

```python
from recipe_optimize import recipe_optimize, load_aafco_constraints
from data_utils import get_all_allergic_fdc_ids

# 获取过敏食物ID
allergy_foods = ['pork', 'chocolate']
allergy_food_ids = list(get_all_allergic_fdc_ids(allergy_foods))

# 加载营养需求
nut_requirements = load_aafco_constraints('adult')

# 优化食谱
result = recipe_optimize(
    nut_requirements=nut_requirements,
    target_energy=1500,
    min_variety=4,
    max_variety=8,
    allergy_foods_id=allergy_food_ids
)

# 处理结果
if result['status'] in ['Optimal', 'Feasible']:
    for food in result['selected_foods']:
        print(f"{food[1]}: {food[2]}g")
```

## 数据准备

本工具需要以下数据文件:

1. `food.csv`: 包含食物ID、描述和类别
2. `food_nutrient.csv`: 食物与营养素的关联
3. `nutrient.csv`: 营养素ID、名称和单位

您可以使用`data_utils`模块处理原始数据:

```python
from data_utils import merge_food_data, add_food_category_labels, export_food_data_stats

# 合并原始数据生成food_data.csv
merge_food_data()

# 添加食物分类标签
add_food_category_labels(protein_target, subprotein_target, vegetable_target, 
                       fruits_target, fat_target, carb_target, supplement_target)

# 导出数据统计信息
export_food_data_stats()
```

## 配置

本工具使用`config.ini`文件存储配置参数:

```ini
[FoodCategoryRatio]
protein_min = 0.30
protein_max = 0.40
subprotein_min = 0.05
subprotein_max = 0.10
# 更多配置...

[FoodCategoryCount]
protein_min = 1
protein_max = 2
# 更多配置...

[CaToP]
min = 1.0
max = 2.0

[Optimization]
default_alpha = 100.0
default_beta = 10.0
# 更多配置...
```

首次运行时会自动创建默认配置，您可以根据需要修改。

## API服务(可选)

如果需要将工具作为API服务提供，可以使用FastAPI实现：

```bash
# 安装API依赖
pip install fastapi uvicorn

# 启动API服务
uvicorn api:app --host 0.0.0.0 --port 8000
```

然后可以通过HTTP请求使用：

```bash
curl -X POST "http://localhost:8000/api/recipe/optimize" \
     -H "Content-Type: application/json" \
     -d '{"target_energy": 1500, "allergy_foods_id": [123, 456]}'
```

# README.md - 快速开始指南
# 宠物食谱优化API服务

这是一个基于Docker的宠物食谱优化API服务，提供了以下功能：

- 基于营养需求生成平衡的宠物食谱
- 考虑食物类别约束和过敏限制
- 支持同步和异步API调用
- 内置安全措施和性能优化

## 快速开始

1. 克隆代码库并进入目录
```bash
git clone https://github.com/yourusername/pet-recipe-api.git
cd pet-recipe-api
```

2. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，设置API密钥和其他选项
```

3. 启动服务
```bash
docker-compose up -d
```

4. 验证服务
```bash
curl http://localhost:8000/api/health
```

## API使用示例

### 优化食谱（同步）

```bash
curl -X POST "http://localhost:8000/api/recipe/optimize" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key-1" \
     -d '{
       "target_energy": 1500,
       "min_variety": 4,
       "max_variety": 8,
       "allergy_foods": ["pork", "chocolate"],
       "stage": "adult"
     }'
```

### 优化食谱（异步）

```bash
curl -X POST "http://localhost:8000/api/recipe/optimize-async" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key-1" \
     -d '{
       "target_energy": 1500,
       "allergy_foods": ["pork", "chocolate"],
       "stage": "adult"
     }'
```

### 获取结果

```bash
curl "http://localhost:8000/api/recipe/result/{result_id}" \
     -H "X-API-Key: your-api-key-1"
```

## 安全说明

- 使用HTTPS进行所有API通信
- 确保API密钥保密性
- 遵循最小权限原则
- 定期更换API密钥

## 日志和监控

- API日志位于: logs/api.log
- Nginx日志位于: nginx/logs/
- 使用健康检查端点监控服务状态

## 生产环境优化

1. 调整配置
```bash
# 增加工作进程
WORKERS=8

# 调整限流设置
RATE_LIMIT=200
```

2. 启用HTTPS
- 配置SSL证书
- 将证书文件放在 `nginx/ssl/` 目录下
- 更新 `nginx/conf.d/default.conf` 中的服务器名称

3. 设置持久化存储
- 使用外部数据库或文件存储服务
- 配置定期备份策略

4. 资源扩展
- 增加容器资源限制
- 设置自动扩缩容策略

## 注意事项

- 本工具假设所有食物数据都是基于100克的标准量计算的
- 营养素需求是基于每1000千卡能量的标准，会按实际目标能量自动缩放
- 优化权重(`alpha`, `beta`, `variety_factor`)影响结果，可能需要调整

## 许可

此项目采用MIT许可证
