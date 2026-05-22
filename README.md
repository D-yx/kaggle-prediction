# 当前最好（越低越好，更新请及时修改）

##### 0.12269->0.12183->0.12157

# 代码简述

当前最优模型：stacking方案，岭回归学习器学习XGBoost、LightGBM、Lasso、ElasticNet、Ridge五种模型的最优组合

新增两个安全的邻域均值特征 ：Neighborhood\_OverallQual（每个邻域的 OverallQual 均值）Neighborhood\_YearBuilt（每个邻域的 YearBuilt 均值）

XGBoost/LightGBM 的树数量从 3000 减至 1500，reg\_alpha 从 0.005 增至 0.01，reg\_lambda 从 1.0 增至 1.5，并加入 early\_stopping\_rounds=50，

在交叉验证 fold 内使用验证集早停，抑制过拟合

减弱元模型正则化，额外加了一个基模型 get\_gbdt\_model()

# 运行代码

库下载  
pip install pandas numpy scikit-learn xgboost lightgbm catboost mlxtend

进入项目根目录  
cd kaggle-prediction

运行预测代码  
python src/predict.py

stacking方案：运行stacking\_method/house\_prices\_solution.py

enhanced\_stacking方案：python enhanced\_stacking/enhanced\_stacking.py



# 协作方式

1、Fork目标项目  
在 GitHub 上把它fork到你自己的账号下  
2、将项目Clone到自己电脑  
3、进入项目并创建新分支  
4、保存并提交（Commit）你的修改  
5、推送到（Push）你的 GitHub 仓库  
6、发起 Pull Request (PR)，把修改请求发到我的仓库来

