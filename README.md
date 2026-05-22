# 当前最好（越低越好，更新请及时修改）
0.11878

# 代码简述
当前最优模型：stacking方案，岭回归学习器学习XGBoost、LightGBM、Lasso、ElasticNet、Ridge五种模型的最优组合
新增两个安全的邻域均值特征
Neighborhood_OverallQual：每个邻域的平均 OverallQual
Neighborhood_YearBuilt：每个邻域的平均 YearBuilt
这些特征通过分组均值计算，不涉及目标编码（无数据泄露），能够为模型提供地理位置相关的全局信息
增加互信息特征选择
使用 SelectKBest(mutual_info_regression, k=200) 从所有特征中筛选出与目标变量关联最强的 200 个特征。
使用三个不同的随机种子（42, 123, 456）分别训练完整的 Stacking 模型，将三个模型的预测结果取平均值。

# 运行代码
库下载  
pip install pandas numpy scikit-learn xgboost lightgbm catboost mlxtend  

进入项目根目录  
cd kaggle-prediction  

运行预测代码  
python src/predict.py  

stacking方案：运行stacking_method/house_prices_solution.py


# 协作方式
1、Fork目标项目  
    在 GitHub 上把它fork到你自己的账号下  
2、将项目Clone到自己电脑  
3、进入项目并创建新分支  
4、保存并提交（Commit）你的修改  
5、推送到（Push）你的 GitHub 仓库  
6、发起 Pull Request (PR)，把修改请求发到我的仓库来  
