# 当前最好（越低越好，更新请及时修改）
0.11798

# 代码简述
当前最优模型：增强版 stacking + 单模型 blending。

Stacking 基模型包括 Lasso、ElasticNet、KernelRidge、GradientBoostingRegressor、XGBoost、CatBoost，元模型为 LightGBM。使用三个不同随机种子（42, 123, 456）分别训练完整 Stacking 模型，并对预测结果取平均。

特征处理保留 Neighborhood_OverallQual，删除实验中表现为噪声的 Neighborhood_YearBuilt；新增 TotalBath 和 OverallQual_TotalSF 两个有效人工特征。继续使用 SelectKBest(mutual_info_regression, k=200) 做互信息特征选择，实际特征数由 One-Hot 和 VarianceThreshold(threshold=0.01) 后的特征数决定。

最终预测在 Stacking 多种子平均结果基础上，额外融合单独训练的 XGBoost 和 CatBoost。当前默认最终融合权重为 Stacking 0.80、XGBoost 0.15、CatBoost 0.05。

# 实验记录

## 有效改动

| 实验 | Kaggle 分数 | 结论 |
|---|---:|---|
| 更换随机种子逻辑后的原方案 | 0.11885 | 后续实验基准 |
| 新增 TotalBath + OverallQual_TotalSF | 0.11868 | 有效，保留 |
| 删除 Neighborhood_YearBuilt | 0.11851 | 有效，该特征为噪声 |
| 加入最终 blending：Stacking 0.85 / XGBoost 0.10 / CatBoost 0.05 | 0.11806 | 明显有效 |
| 调整 blending：Stacking 0.80 / XGBoost 0.15 / CatBoost 0.05 | 0.11798 | 当前最好 |

## 负收益或未采用实验

| 实验 | Kaggle 分数 | 结论 |
|---|---:|---|
| 一次性加入 11 个人工特征 | 0.11902 | 整体变差，说明部分特征引入噪声 |
| 在有效特征基础上加入 OverallQual_GrLivArea | 0.11875 | 变差，删除 |
| 在有效特征基础上加入 HouseAge | 0.11881 | 变差，删除 |
| 删除 MiscVal + PoolArea + 3SsnPorch + LowQualFinSF | 0.11927 | 变差，恢复 |
| 删除 Neighborhood_OverallQual | 0.11918 | 明显变差，说明该特征有效 |
| 删除 RoofMatl_CompShg | 0.11878 | 变差，恢复 |
| 删除 Heating_GasA | 0.11909 | 变差，恢复 |
| 删除 KitchenAbvGr | 0.11904 | 变差，恢复 |
| VarianceThreshold = 0.001 | 约 0.12 | 放回低频特征过多，变差 |
| VarianceThreshold = 0.005 | 约 0.12 | 变差 |
| VarianceThreshold = 0.02 | 0.11942 | 删掉低频特征过多，变差 |

# 运行代码
库下载  
pip install pandas numpy scipy scikit-learn xgboost lightgbm catboost mlxtend  

进入项目根目录  
cd kaggle-prediction  

运行当前最优预测代码  
python 0.11798/enhanced_stacking.py  

当前默认最终融合权重：Stacking 0.80、XGBoost 0.15、CatBoost 0.05。可通过 --stacking-weight、--xgb-weight、--catboost-weight 调整。


代码已使用脚本文件位置自动定位项目根目录和 data 目录，因此在 PyCharm 中运行时不再依赖 Working directory。


# 协作方式
1、Fork目标项目  
    在 GitHub 上把它fork到你自己的账号下  
2、将项目Clone到自己电脑  
3、进入项目并创建新分支  
4、保存并提交（Commit）你的修改  
5、推送到（Push）你的 GitHub 仓库  
6、发起 Pull Request (PR)，把修改请求发到我的仓库来  
