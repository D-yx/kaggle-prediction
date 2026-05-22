"""
Kaggle House Prices: 稳健提升版
- 新增少量有效交互特征（无目标编码）
- 降低树模型迭代次数 + 早停
- 元模型 Ridge(alpha=0.5)
- 可选加入 CatBoost
"""

import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")

from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# ==================== 数据加载与清洗 ====================
def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_id = train["Id"]
    test_id = test["Id"]
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    return train, test, train_id, test_id

def remove_outliers(train):
    outlier_idx = train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index
    train = train.drop(outlier_idx).reset_index(drop=True)
    return train

def target_transform(train):
    y_train = np.log1p(train["SalePrice"])
    train.drop("SalePrice", axis=1, inplace=True)
    return train, y_train

def handle_missing_values(all_data):
    # 车库相关
    for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        all_data[col] = all_data[col].fillna("None")
    for col in ["GarageYrBlt", "GarageArea", "GarageCars"]:
        all_data[col] = all_data[col].fillna(0)
    # 地下室相关
    for col in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
        all_data[col] = all_data[col].fillna("None")
    for col in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                "BsmtFullBath", "BsmtHalfBath"]:
        all_data[col] = all_data[col].fillna(0)
    # 其他
    for col in ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]:
        all_data[col] = all_data[col].fillna("None")
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ["MSZoning", "Electrical", "KitchenQual", "Exterior1st",
                "Exterior2nd", "SaleType", "Functional"]:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    all_data.drop(["Utilities"], axis=1, inplace=True)
    return all_data

def feature_engineering(all_data):
    # 基础组合
    all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    all_data["TotalPorchSF"] = (all_data["OpenPorchSF"] + all_data["EnclosedPorch"] +
                                all_data["3SsnPorch"] + all_data["ScreenPorch"] + all_data["WoodDeckSF"])
    all_data["TotalBathrooms"] = (all_data["FullBath"] + 0.5 * all_data["HalfBath"] +
                                  all_data["BsmtFullBath"] + 0.5 * all_data["BsmtHalfBath"])
    # 二值特征
    all_data["HasPool"] = (all_data["PoolArea"] > 0).astype(int)
    all_data["Has2ndFloor"] = (all_data["2ndFlrSF"] > 0).astype(int)
    all_data["HasGarage"] = (all_data["GarageArea"] > 0).astype(int)
    all_data["HasBsmt"] = (all_data["TotalBsmtSF"] > 0).astype(int)
    all_data["HasFireplace"] = (all_data["Fireplaces"] > 0).astype(int)
    # 时间特征
    all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
    all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]
    all_data["IsNewHouse"] = (all_data["YearBuilt"] == all_data["YrSold"]).astype(int)
    # 质量×面积交互
    all_data["OverallQual_TotalSF"] = all_data["OverallQual"] * all_data["TotalSF"]
    all_data["OverallQual_GrLivArea"] = all_data["OverallQual"] * all_data["GrLivArea"]

    # 新增有效特征（无目标泄漏，无类型错误）
    all_data["Neighborhood_OverallQual"] = all_data.groupby("Neighborhood")["OverallQual"].transform("mean")
    all_data["Neighborhood_OverallQual"] = all_data["Neighborhood_OverallQual"].fillna(all_data["OverallQual"].mean())

    all_data["Neighborhood_YearBuilt"] = all_data.groupby("Neighborhood")["YearBuilt"].transform("mean")
    all_data["Neighborhood_YearBuilt"] = all_data["Neighborhood_YearBuilt"].fillna(all_data["YearBuilt"].mean())

    return all_data

def feature_transformation(all_data):
    ordinal_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, "None":0}
    ordinal_cols = ["ExterQual","ExterCond","BsmtQual","BsmtCond",
                    "HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]
    for col in ordinal_cols:
        all_data[col] = all_data[col].map(ordinal_map).fillna(0).astype(int)
    bsmt_exposure_map = {"Gd":4, "Av":3, "Mn":2, "No":1, "None":0}
    all_data["BsmtExposure"] = all_data["BsmtExposure"].map(bsmt_exposure_map).fillna(0).astype(int)
    bsmt_fin_map = {"GLQ":6, "ALQ":5, "BLQ":4, "Rec":3, "LwQ":2, "Unf":1, "None":0}
    for col in ["BsmtFinType1","BsmtFinType2"]:
        all_data[col] = all_data[col].map(bsmt_fin_map).fillna(0).astype(int)
    garage_finish_map = {"Fin":3, "RFn":2, "Unf":1, "None":0}
    all_data["GarageFinish"] = all_data["GarageFinish"].map(garage_finish_map).fillna(0).astype(int)
    fence_map = {"GdPrv":4, "MnPrv":3, "GdWo":2, "MnWw":1, "None":0}
    all_data["Fence"] = all_data["Fence"].map(fence_map).fillna(0).astype(int)
    functional_map = {"Typ":7, "Min1":6, "Min2":5, "Mod":4, "Maj1":3, "Maj2":2, "Sev":1, "Sal":0}
    all_data["Functional"] = all_data["Functional"].map(functional_map).fillna(7).astype(int)
    paved_map = {"Y":2, "P":1, "N":0}
    all_data["PavedDrive"] = all_data["PavedDrive"].map(paved_map).fillna(0).astype(int)
    all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)

    numeric_feats = all_data.select_dtypes(include=[np.number]).columns
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index
    lam = 0.15
    for feat in skewed_feats:
        all_data[feat] = boxcox1p(all_data[feat].astype(float), lam)

    all_data = pd.get_dummies(all_data)
    return all_data

# ==================== 模型定义（减少树数量，保持早停） ====================
def get_xgboost_model():
    return xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.01, max_depth=4,
        min_child_weight=3, gamma=0.0, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.01, reg_lambda=1.5, objective="reg:squarederror",
        early_stopping_rounds=50, eval_metric='rmse',
        n_jobs=-1, random_state=42
    )

def get_lightgbm_model():
    return lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.01, num_leaves=31,
        max_depth=-1, min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.01, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1,
        early_stopping_rounds=50
    )

def get_gbdt_model():
    return GradientBoostingRegressor(
        n_estimators=1500, learning_rate=0.01, max_depth=4,
        max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
        loss="huber", random_state=42
    )

def get_catboost_model():
    return CatBoostRegressor(
        iterations=1000, learning_rate=0.01, depth=4,
        l2_leaf_reg=3, eval_metric='RMSE', early_stopping_rounds=50,
        random_seed=42, logging_level='Silent'
    )

def get_lasso_model():
    return make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))

def get_elasticnet_model():
    return make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42))

def get_ridge_model():
    return make_pipeline(RobustScaler(), Ridge(alpha=10.0))

# ==================== Stacking 集成器（支持早停） ====================
class StackingAveragedModels(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        X = np.array(X)
        y = np.array(y)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                X_train_fold, X_val_fold = X[train_index], X[holdout_index]
                y_train_fold, y_val_fold = y[train_index], y[holdout_index]
                instance = clone(model)
                if hasattr(instance, 'early_stopping_rounds'):
                    try:
                        instance.fit(
                            X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)]
                        )
                    except Exception:
                        instance.fit(X_train_fold, y_train_fold)
                else:
                    instance.fit(X_train_fold, y_train_fold)
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X_val_fold)
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

def build_stacking_model():
    base_models = [
        get_xgboost_model(),
        get_lightgbm_model(),
        get_gbdt_model(),
        get_catboost_model(),   # 可选，确保已安装 catboost
        get_lasso_model(),
        get_elasticnet_model(),
        get_ridge_model(),
    ]
    # 元模型 Ridge 正则化更弱
    meta_model = Ridge(alpha=0.5)
    return StackingAveragedModels(base_models=base_models, meta_model=meta_model, n_folds=5)

def rmsle_cv(model, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("加载数据...")
    train, test, train_id, test_id = load_data()
    ntrain = train.shape[0]
    print(f"训练集: {train.shape}, 测试集: {test.shape}")

    print("\n数据清洗...")
    train = remove_outliers(train)
    train, y_train = target_transform(train)
    ntrain = train.shape[0]
    print(f"移除离群值后训练集: {train.shape}")

    print("\n特征工程...")
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    all_data = handle_missing_values(all_data)
    all_data = feature_engineering(all_data)
    all_data = feature_transformation(all_data)

    X_train = all_data[:ntrain].values.astype(np.float64)
    X_test = all_data[ntrain:].values.astype(np.float64)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # 移除零方差特征
    variances = np.var(X_train, axis=0)
    keep = variances > 0
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    print(f"特征总数: {X_train.shape[1]}")

    print("\n训练 Stacking 模型（元模型=Ridge, alpha=0.5, 带早停）...")
    stacking = build_stacking_model()
    stacking_score = rmsle_cv(stacking, X_train, y_train)
    print(f"5折交叉验证 RMSLE: {stacking_score.mean():.5f} (+/- {stacking_score.std():.5f})")

    print("全量训练并预测测试集...")
    stacking.fit(X_train, y_train)
    final_pred = stacking.predict(X_test)
    final_pred = np.clip(final_pred, 9.0, 14.0)

    submission = pd.DataFrame({"Id": test_id, "SalePrice": np.expm1(final_pred)})
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "submission_boost.csv")
    submission.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\n提交文件已保存: {output_path}")
    print(f"预测价格范围: ${submission['SalePrice'].min():,.0f} ~ ${submission['SalePrice'].max():,.0f}")
    print(f"预测价格均值: ${submission['SalePrice'].mean():,.0f}")

if __name__ == "__main__":
    main()