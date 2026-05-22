"""
Kaggle House Prices: 增强版 Stacking（优化版，修正元模型早停错误）
- 基模型：Lasso, ElasticNet, KernelRidge, GBDT, XGBoost, CatBoost
- 元模型：LightGBM（无早停，减少树数量）
- 新增：邻域均值特征 + 互信息特征选择 + 多种子平均
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

SEEDS = [42, 123, 456]  # 多种子平均


# ---------------------- 1. 加载数据 ----------------------
def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test


# ---------------------- 2. 数据预处理（增加邻域特征） ----------------------
def preprocess_data(train, test):
    # 移除离群点
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

    # 保存目标变量并移除
    y_train = np.log1p(train['SalePrice'])
    train = train.drop('SalePrice', axis=1)

    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    all_data.drop('Id', axis=1, inplace=True)

    # 缺失值处理（与原代码相同）
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')

    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    # 新增：邻域均值特征（安全）
    all_data["Neighborhood_OverallQual"] = all_data.groupby("Neighborhood")["OverallQual"].transform("mean")
    all_data["Neighborhood_OverallQual"] = all_data["Neighborhood_OverallQual"].fillna(all_data["OverallQual"].mean())
    all_data["Neighborhood_YearBuilt"] = all_data.groupby("Neighborhood")["YearBuilt"].transform("mean")
    all_data["Neighborhood_YearBuilt"] = all_data["Neighborhood_YearBuilt"].fillna(all_data["YearBuilt"].mean())

    return all_data, y_train, train, test


# ---------------------- 3. 特征工程 ----------------------
def feature_engineering(all_data):
    # 类型转换与标签编码
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    ordinal_cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                    'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
                    'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                    'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
                    'YrSold', 'MoSold')
    for c in ordinal_cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))

    # 面积组合
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    # 偏度校正
    numeric_feats = all_data.select_dtypes(include=[np.number]).columns
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index
    lam = 0.15
    for feat in skewed_feats:
        all_data[feat] = boxcox1p(all_data[feat].astype(float), lam)

    # One-Hot 编码
    all_data = pd.get_dummies(all_data)

    # 方差过滤
    selector = VarianceThreshold(threshold=0.01)
    all_data = selector.fit_transform(all_data)

    return all_data


# ---------------------- 4. 模型定义 ----------------------
def get_base_models():
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))
    elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42))
    kernel_ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    gboost = GradientBoostingRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
        min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42
    )
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=3000, learning_rate=0.01,
        max_depth=4, subsample=0.7, colsample_bytree=0.7, random_state=42,
        early_stopping_rounds=50
    )
    catboost_model = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.01, depth=5, verbose=False,
        random_state=42, early_stopping_rounds=50
    )
    return [lasso, elastic_net, kernel_ridge, gboost, xgb_model, catboost_model]


# ---------------------- 5. Stacking 集成器（支持早停） ----------------------
class StackingAveragedModels(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5, seed=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        X = np.array(X)
        y = np.array(y)
        oof_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                X_train_fold, X_val_fold = X[train_idx], X[holdout_idx]
                y_train_fold, y_val_fold = y[train_idx], y[holdout_idx]

                if hasattr(instance, 'early_stopping_rounds'):
                    if isinstance(instance, xgb.XGBRegressor):
                        instance.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
                    elif isinstance(instance, cb.CatBoostRegressor):
                        instance.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), verbose=False)
                    else:
                        instance.fit(X_train_fold, y_train_fold)
                else:
                    instance.fit(X_train_fold, y_train_fold)

                self.base_models_[i].append(instance)
                oof_predictions[holdout_idx, i] = instance.predict(X_val_fold)

        self.meta_model_.fit(oof_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)


# ---------------------- 6. 主流程（多种子平均 + 互信息特征选择） ----------------------
def main():
    print("加载数据...")
    train, test = load_data()

    print("数据预处理...")
    all_data, y_train, train, test = preprocess_data(train, test)

    print("特征工程...")
    all_data = feature_engineering(all_data)

    # 分割训练/测试
    X_train = all_data[:len(train)].astype(np.float64)
    X_test = all_data[len(train):].astype(np.float64)

    # 互信息特征选择（保留200个特征）
    print("互信息特征选择（k=200）...")
    selector = SelectKBest(mutual_info_regression, k=min(200, X_train.shape[1]))
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    print(f"最终特征数: {X_train.shape[1]}")

    # 多种子训练
    predictions = []
    for seed in SEEDS:
        print(f"\n训练种子: {seed}")
        # 修正：移除 early_stopping_rounds，减少 n_estimators
        meta_model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=1000,  # 减少树数量
            learning_rate=0.01,
            max_depth=5,
            min_child_samples=15,
            min_split_gain=0.01,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=seed,
            verbose=-1
            # 去掉了 early_stopping_rounds
        )
        stacking = StackingAveragedModels(
            base_models=get_base_models(),
            meta_model=meta_model,
            n_folds=5,
            seed=seed
        )
        stacking.fit(X_train, y_train)
        pred = stacking.predict(X_test)
        predictions.append(pred)

    # 平均预测
    final_pred_log = np.mean(predictions, axis=0)
    final_pred_log = np.clip(final_pred_log, 9.0, 14.0)
    submission = pd.DataFrame({'Id': test['Id'].values, 'SalePrice': np.expm1(final_pred_log)})
    submission.to_csv('enhanced_stacking/submission_optimized.csv', index=False, float_format='%.6f')
    print("\n✅ submission_optimized.csv 已保存！")
    print(f"预测价格范围: ${submission['SalePrice'].min():,.0f} ~ ${submission['SalePrice'].max():,.0f}")
    print(f"预测价格均值: ${submission['SalePrice'].mean():,.0f}")


if __name__ == '__main__':
    main()