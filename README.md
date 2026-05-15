# 自助量贩餐厅菜量需求预测与运营优化设计

本目录用于整理数学建模赛题 B《自助量贩餐厅菜量需求预测与运营优化设计》的数据、代码、结果和论文材料。

## 目录说明

```text
.
├── README.md                         项目总说明
├── 附件1餐厅销售流水信息表.xlsx         原始附件 1
├── 附件2部分消费订单菜品具体信息表.xlsx   原始附件 2
├── 2026长三角赛题B...pdf              赛题原文
├── *.csv                            公共清洗数据与汇总数据
├── problem2_*.py / problem3_*.py    根目录工作脚本
├── q4_package_optimization.py       问题四工作脚本
├── figures/                         工作图表输出
├── results/                         问题四工作结果输出
├── 第二题/                          第二题正式成果
├── 第三题/                          第三题正式成果
├── 问题四/                          问题四正式成果与写作材料
└── 归档_根目录旧副本/                 从根目录移出的重复结果副本
```

## 文件组织原则

- 根目录保留公共数据和可执行脚本，便于各题脚本按原有相对路径运行。
- `第二题/`、`第三题/`、`问题四/` 作为各题的正式成果目录，写论文时优先从这些文件夹取材料。
- `figures/` 和 `results/` 是脚本运行时生成的工作输出目录。
- `归档_根目录旧副本/` 保存曾经散落在根目录的重复结果文件，避免误删，同时让根目录更清爽。

## 公共数据文件

### 原始数据

- `附件1餐厅销售流水信息表.xlsx`
- `附件2部分消费订单菜品具体信息表.xlsx`

### 主要处理数据

- `flow_cleaned.csv`：清洗后的流水数据
- `detail_cleaned.csv`：清洗后的消费菜品明细
- `merged_order_dish.csv`：订单与菜品明细合并表
- `order_summary.csv`：订单层汇总
- `meal_summary.csv`：餐次层汇总
- `daily_summary.csv`：日度汇总
- `dish_summary.csv`：菜品汇总
- `dish_summary_with_abc.csv`：带 ABC 分类的菜品汇总
- `basket.csv`：订单篮子数据

## 各题成果入口

### 第二题

目录：`第二题/`

主要文件：

- `problem2_all_methods.py`：第二题主脚本
- `problem2_prediction_results.xlsx`：预测结果
- `problem2_model_evaluation.xlsx`：模型评价
- `problem2_paper_document.md`：论文正文材料
- `problem2_summary.md`：结果摘要
- `figures/`：第二题论文图

### 第三题

目录：`第三题/`

主要文件：

- `problem3_lunch_optimization.py`：多目标整数规划方案
- `problem3_nsga2_lunch_optimization.py`：NSGA-II 方案
- `problem3_greedy_lunch_optimization.py`：贪心基准方案
- `problem3_compare_methods.py`：方法比较
- `problem3_paper_document.md`：论文正文材料
- `problem3_summary.md`：结果摘要
- `figures/`：第三题论文图

### 问题四

目录：`问题四/`

主要文件：

- `q4_package_optimization.py`：问题四套餐组合优化脚本
- `problem4_paper_reference.md`：完整论文参考稿
- `problem4_outline.md`：写作提纲
- `results/q4_nsga2_packages.csv`：NSGA-II 推荐套餐
- `results/q4_greedy_packages.csv`：贪心基准套餐
- `results/q4_method_comparison.csv`：主模型与基准模型对比
- `results/q4_latex_tables.txt`：可直接引用的 LaTeX 表格
- `results/q4_model_description.txt`：问题四模型说明文字
- `figures/`：问题四图表

## 建议使用方式

### 写论文时

优先查看：

1. `第二题/problem2_paper_document.md`
2. `第三题/problem3_paper_document.md`
3. `问题四/problem4_paper_reference.md`
4. 各题目录中的 `figures/`、`results/` 和 LaTeX 表格文件

### 重新运行脚本时

建议在项目根目录执行，因为当前脚本大多按根目录相对路径读取公共数据。

例如：

```bash
/Users/linjiamin/venv/bin/python problem2_all_methods.py
/Users/linjiamin/venv/bin/python problem3_lunch_optimization.py
/Users/linjiamin/venv/bin/python q4_package_optimization.py
```

## 当前整理状态

- 根目录的公共数据保留；
- 第二题、第三题的重复结果文件已移入 `归档_根目录旧副本/`；
- 第二题、第三题、问题四均已有独立成果目录；
- 问题四新增了论文参考稿和写作提纲，便于直接改写入正文。

## 维护建议

- 新生成的正式成果优先放入对应题目文件夹；
- 根目录只保留公共数据、工作脚本和必要工作输出；
- 若再次重跑脚本产生根目录结果副本，可在确认题目文件夹已同步后移入归档；
- 论文最终引用图片时，尽量统一从各题文件夹下的 `figures/` 取图，避免路径混乱。
