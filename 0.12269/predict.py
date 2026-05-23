import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LassoCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from scipy.special import boxcox1p

warnings.filterwarnings('ignore')



def preprocess_ames_data(train_path='../data/train.csv', test_path='../data/test.csv'):
    """
    读取并执行Ames房价数据的全套预处理工作。
    返回处理好的 X_train, y_train, 和 X_test
    """
    print("正在加载数据...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 1. 异常值剔除 (Outliers)
    # 根据原作者的建议，剔除面积极大但价格极低的离群点
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index).reset_index(drop=True)

    # 2. 提取目标变量并合并特征
    # Kaggle 评分标准是基于对数误差的，因此这里使用 log1p 
    y_train = np.log1p(train['SalePrice'])
    
    # 保存 Id 用于最后生成提交文件
    test_id = test['Id']
    
    # 删除 Id 和目标变量，将 train 和 test 合并以便统一处理特征
    train_features = train.drop(['Id', 'SalePrice'], axis=1)
    test_features = test.drop(['Id'], axis=1)
    features = pd.concat([train_features, test_features]).reset_index(drop=True)

    print("正在处理缺失值...")
    # 3. 填补缺失值 (NA)
    # 类别1：NA代表"没有该设施" (根据数据字典)
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        features[col] = features[col].fillna('None')

    # 类别2：NA代表 0 的数值特征 (没有车库/地下室，面积和数量自然是0)
    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        features[col] = features[col].fillna(0)

    # 类别3：特殊填补 - 街道连接长度 (LotFrontage)
    # 假设同一街区 (Neighborhood) 的房屋临街距离相近，使用该街区的均值填补
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    # 类别4：使用众数填补 (基础特征少量缺失)
    mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional']
    for col in mode_cols:
        features[col] = features[col].fillna(features[col].mode()[0])

    # 类别5：无区分度特征剔除 (Utilities 几乎全部是 AllPub，只有一个 NoSeWa)
    features = features.drop(['Utilities'], axis=1)

    print("正在进行特征工程...")
    # 4. 数据类型转换 (将数值型的类别特征转化为字符型)
    features['MSSubClass'] = features['MSSubClass'].apply(str)
    features['OverallCond'] = features['OverallCond'].apply(str)
    features['YrSold'] = features['YrSold'].apply(str)
    features['MoSold'] = features['MoSold'].apply(str)

    # 5. 核心特征工程 (Feature Engineering)
    # 房屋总面积 (地下室 + 一楼 + 二楼) - 对房价影响最大的单一特征
    features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
    
    # 房屋总卫浴数
    features['TotalBath'] = features['FullBath'] + (0.5 * features['HalfBath']) + \
                            features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath'])
    
    # 布尔值特征：是否拥有地下室、二楼、壁炉等
    features['HasBsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    # 6. 特征编码 (Label Encoding)
    # 对于带有顺序性质的离散特征，使用 Label Encoder 保留其次序关系
    cols_to_encode = [
        'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold'
    ]
    for col in cols_to_encode:
        lbl = LabelEncoder()
        lbl.fit(list(features[col].values))
        features[col] = lbl.transform(list(features[col].values))

    # 7. 处理数值特征的偏度 (Skewness)
    # 这对 Lasso/Ridge 等线性模型至关重要
    numeric_feats = features.select_dtypes(include=[np.number]).columns
    skewed_feats = features[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness_df = pd.DataFrame({'Skew' :skewed_feats})
    
    # 筛选出偏度绝对值大于 0.75 的特征
    skewed_features_to_fix = skewness_df[abs(skewness_df['Skew']) > 0.75].index
    
    # 使用 Box-Cox 转换使数据更接近正态分布
    lam = 0.15
    for feat in skewed_features_to_fix:
        features[feat] = boxcox1p(features[feat], lam)

    # 8. 独热编码 (One-Hot Encoding)
    # 处理剩下的无序离散特征
    features = pd.get_dummies(features)
    print(f"数据预处理完成，特征总维度: {features.shape[1]}")

    # 9. 分割回训练集和测试集
    X_train = features.iloc[:len(y_train), :]
    X_test = features.iloc[len(y_train):, :]

    return X_train, y_train, X_test, test_id


# 1. 调用预处理函数获取完全干净、处理好的 Numpy 矩阵和测试集ID
print("步骤 1: 数据预处理...")
X_train_df, y_train_series, X_test_df, test_id = preprocess_ames_data('../data/train.csv', '../data/test.csv')

# 将 DataFrame 转换为模型所需的 Numpy 数组
X_train = np.array(X_train_df)
y_train = np.array(y_train_series)
X_test = np.array(X_test_df)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}\n")


# 2. 定义高级机器学习模型
print("步骤 2: 初始化基础模型与融合模型...")
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# 模型1：XGBoost
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006, random_state=42)

# 模型2：LightGBM
lightgbm = LGBMRegressor(objective='regression', 
                         num_leaves=4,
                         learning_rate=0.01, 
                         n_estimators=5000,
                         max_bin=200, 
                         bagging_fraction=0.75,
                         bagging_freq=5, 
                         bagging_seed=7,
                         feature_fraction=0.2,
                         feature_fraction_seed=7,
                         verbose=-1,
                         random_state=42)

# 模型3：CatBoost
catboost = CatBoostRegressor(iterations=6000,
                             learning_rate=0.005,
                             depth=4,
                             l2_leaf_reg=1,
                             eval_metric='RMSE',
                             random_seed=42,
                             logging_level='Silent')

# 模型4：正则化线性模型
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=[14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=int(1e7), alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006], random_state=42, cv=kfolds))

# 构建 Stacking 融合模型
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, catboost, ridge, lasso),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# 3. 训练与预测
print("\n步骤 3: 开始训练模型 (这可能需要几分钟时间)...")
print("--> 训练 XGBoost...")
xgboost.fit(X_train, y_train)

print("--> 训练 LightGBM...")
lightgbm.fit(X_train, y_train)

print("--> 训练 CatBoost...")
catboost.fit(X_train, y_train)

print("--> 训练 Ridge...")
ridge.fit(X_train, y_train)

print("--> 训练 Lasso...")
lasso.fit(X_train, y_train)

print("--> 训练 Stacking 融合网络...")
stack_gen.fit(X_train, y_train)


# 4. 定义模型加权融合预测函数
def blend_models_predict(X):
    return ((0.1 * ridge.predict(X)) + 
            (0.15 * lasso.predict(X)) + 
            (0.2 * xgboost.predict(X)) + 
            (0.15 * lightgbm.predict(X)) + 
            (0.1 * catboost.predict(X)) + 
            (0.3 * stack_gen.predict(X)))

print("\n步骤 4: 对测试集进行预测...")
# 调用融合函数，并使用 np.expm1 还原通过 log1p 对数化处理过的房价
preds = np.expm1(blend_models_predict(X_test))


# 5. 生成结果文件并保存到指定文件夹
print("\n步骤 5: 生成提交文件...")
# 创建 result 文件夹（如果不存在）
os.makedirs('result', exist_ok=True)

# 构建符合 Kaggle 样例的数据框
submission = pd.DataFrame({
    'Id': test_id,
    'SalePrice': preds
})

# 保存为 CSV 文件，不包含索引列
save_path = os.path.join('result', 'submission.csv')
submission.to_csv(save_path, index=False)

print(f"预测完成！提交文件已成功保存至: {save_path}")