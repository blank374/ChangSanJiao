# 附录核心代码说明

本文件夹用于集中存放论文附录中建议保留的核心建模代码。筛选原则是：

1. 能体现主要模型构建、求解与结果比较过程；
2. 与论文正文中的方法描述直接对应；
3. 不包含单纯用于绘图、排版或重复归档的辅助脚本。

## 文件清单

### 第二题

- `第二题/problem2_all_methods.py`
  - 用途：完成需求预测建模、特征构造、模型比较、最终预测和区间估计；
  - 建议：作为第二题附录主代码。

### 第三题

- `第三题/problem3_lunch_optimization.py`
  - 用途：多目标整数规划主模型；
  - 建议：作为第三题主模型代码。
- `第三题/problem3_nsga2_lunch_optimization.py`
  - 用途：NSGA-II 改进模型；
  - 建议：用于体现多目标进化算法对照。
- `第三题/problem3_greedy_lunch_optimization.py`
  - 用途：贪心基准模型；
  - 建议：用于体现基准方案。
- `第三题/problem3_compare_methods.py`
  - 用途：三类方法的结果比较与综合评价；
  - 建议：附在第三题算法代码之后。

### 问题四

- `问题四/q4_package_optimization.py`
  - 用途：固定价位套餐的多目标组合优化、Pareto 候选筛选和贪心对照；
  - 建议：作为问题四附录主代码。

## 推荐附录顺序

1. `problem2_all_methods.py`
2. `problem3_lunch_optimization.py`
3. `problem3_nsga2_lunch_optimization.py`
4. `problem3_greedy_lunch_optimization.py`
5. `problem3_compare_methods.py`
6. `q4_package_optimization.py`

## 未纳入的脚本

以下脚本更偏向图表生成、误差表补充或论文材料整理，通常不必放入核心附录：

- `第二题/draw_problem2_paper_figures.py`
- `第二题/generate_q2_error_evaluation.py`
- `第三题/problem3_profit_nutrition_outputs.py`

如需将“完整可复现代码”而不只是“核心附录代码”打包，可在本文件夹基础上再补入上述辅助脚本。
