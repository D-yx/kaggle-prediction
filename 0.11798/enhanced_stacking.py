"""
Kaggle House Prices: 增强版 Stacking（优化版，修正元模型早停错误）
- 基模型：Lasso, ElasticNet, KernelRidge, GBDT, XGBoost, CatBoost
- 元模型：LightGBM（无早停，减少树数量）
- 新增：邻域均值特征 + 互信息特征选择 + 多种子平均
"""
 
import numpy as np
import pandas as pd
import warnings
import argparse
from pathlib import Path

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
N_FEATURES = 200
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
SUBMISSION_PATH = SCRIPT_DIR / 'submission_optimized.csv'
CV_OOF_PATH = SCRIPT_DIR / 'local_cv_oof.csv'
SELECTED_FEATURES_PATH = SCRIPT_DIR / 'selected_features.csv'


# ---------------------- 1. 加载数据 ----------------------
def load_data():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    return train, test


def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate the enhanced stacking house price model.')
    parser.add_argument('--cv', action='store_true', help='Run local K-Fold CV before training the final submission model.')
    parser.add_argument('--cv-only', action='store_true', help='Run local K-Fold CV and skip final submission training.')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of outer CV folds. Default: 5.')
    parser.add_argument('--n-features', type=int, default=N_FEATURES, help='Number of mutual-information features to keep.')
    parser.add_argument('--list-features', action='store_true', help='Write selected feature names and exit without training.')
    parser.add_argument('--stacking-weight', type=float, default=0.80, help='Final blend weight for stacking predictions.')
    parser.add_argument('--xgb-weight', type=float, default=0.15, help='Final blend weight for single XGBoost predictions.')
    parser.add_argument('--catboost-weight', type=float, default=0.05, help='Final blend weight for single CatBoost predictions.')
    return parser.parse_args()


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

    all_data["Neighborhood_OverallQual"] = all_data.groupby("Neighborhood")["OverallQual"].transform("mean")
    all_data["Neighborhood_OverallQual"] = all_data["Neighborhood_OverallQual"].fillna(all_data["OverallQual"].mean())

    return all_data, y_train, train, test


# ---------------------- 3. 特征工程 ----------------------
def feature_engineering(all_data, return_feature_names=False):
    # 保留低风险的核心组合特征，避免过多冗余特征影响 public LB 表现。
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['TotalBath'] = (
        all_data['FullBath'] +
        0.5 * all_data['HalfBath'] +
        all_data['BsmtFullBath'] +
        0.5 * all_data['BsmtHalfBath']
    )
    all_data['OverallQual_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']

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

    # 偏度校正
    numeric_feats = all_data.select_dtypes(include=[np.number]).columns
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index
    lam = 0.15
    for feat in skewed_feats:
        all_data[feat] = boxcox1p(all_data[feat].astype(float), lam)

    # One-Hot 编码
    all_data = pd.get_dummies(all_data)
    feature_names = all_data.columns.to_numpy()

    # 方差过滤
    selector = VarianceThreshold(threshold=0.01)
    all_data = selector.fit_transform(all_data)
    selected_feature_names = feature_names[selector.get_support()]

    if return_feature_names:
        return all_data, selected_feature_names
    return all_data


def write_selected_features(feature_names):
    features = pd.DataFrame({
        'feature': feature_names,
        'source': [name.split('_', 1)[0] for name in feature_names],
    })
    features.to_csv(SELECTED_FEATURES_PATH, index=False)
    print(f"保留特征数: {len(features)}")
    print(f"特征列表已保存至: {SELECTED_FEATURES_PATH}")


# ---------------------- 4. 模型定义 ----------------------
def get_base_models(seed=42):
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=seed))
    elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=seed))
    kernel_ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    gboost = GradientBoostingRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
        min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=seed
    )
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=3000, learning_rate=0.01,
        max_depth=4, subsample=0.7, colsample_bytree=0.7, random_state=seed,
        early_stopping_rounds=50
    )
    catboost_model = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.01, depth=5, verbose=False,
        random_state=seed, early_stopping_rounds=50
    )
    return [lasso, elastic_net, kernel_ridge, gboost, xgb_model, catboost_model]


def get_meta_model(seed):
    return lgb.LGBMRegressor(
        objective='regression',
        n_estimators=1000,
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
    )


def mutual_info_scores(X, y):
    return mutual_info_regression(X, y, random_state=42)


def select_top_features(X_fit, y_fit, X_apply, n_features=N_FEATURES):
    selector = SelectKBest(mutual_info_scores, k=min(n_features, X_fit.shape[1]))
    X_fit_selected = selector.fit_transform(X_fit, y_fit)
    X_apply_selected = selector.transform(X_apply)
    return X_fit_selected, X_apply_selected, selector


def rmsle_from_log(y_true_log, y_pred_log):
    return np.sqrt(np.mean((np.asarray(y_pred_log) - np.asarray(y_true_log)) ** 2))


def fit_predict_stacking_ensemble(X_train, y_train, X_predict, seeds=SEEDS):
    predictions = []
    for seed in seeds:
        print(f"\n训练种子: {seed}")
        stacking = StackingAveragedModels(
            base_models=get_base_models(seed),
            meta_model=get_meta_model(seed),
            n_folds=5,
            seed=seed
        )
        stacking.fit(X_train, y_train)
        predictions.append(stacking.predict(X_predict))
    return np.mean(predictions, axis=0)


def fit_predict_single_model_blend(X_train, y_train, X_predict, seed=42):
    print("\n训练单模型用于最终 blending: XGBoost")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=3000, learning_rate=0.01,
        max_depth=4, subsample=0.7, colsample_bytree=0.7, random_state=seed
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_predict)

    print("训练单模型用于最终 blending: CatBoost")
    catboost_model = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.01, depth=5, verbose=False,
        random_state=seed
    )
    catboost_model.fit(X_train, y_train)
    catboost_pred = catboost_model.predict(X_predict)

    return xgb_pred, catboost_pred


def run_local_cv(X_train, y_train, train_ids, n_splits=5, n_features=N_FEATURES):
    print("\n本地 K-Fold CV 评估...")
    y_train = np.asarray(y_train)
    train_ids = np.asarray(train_ids)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(X_train.shape[0])
    fold_scores = []

    for fold, (fit_idx, valid_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        print(f"\nCV Fold {fold}/{n_splits}")
        X_fit_raw, X_valid_raw = X_train[fit_idx], X_train[valid_idx]
        y_fit, y_valid = y_train[fit_idx], y_train[valid_idx]

        X_fit, X_valid, _ = select_top_features(X_fit_raw, y_fit, X_valid_raw, n_features)
        print(f"Fold {fold} 特征数: {X_fit.shape[1]}")

        fold_pred = fit_predict_stacking_ensemble(X_fit, y_fit, X_valid)
        fold_pred = np.clip(fold_pred, 9.0, 14.0)
        oof_predictions[valid_idx] = fold_pred

        fold_score = rmsle_from_log(y_valid, fold_pred)
        fold_scores.append(fold_score)
        print(f"Fold {fold} RMSLE: {fold_score:.5f}")

    oof_score = rmsle_from_log(y_train, oof_predictions)
    print("\n本地 CV 结果")
    print(f"Fold RMSLE: {[round(score, 5) for score in fold_scores]}")
    print(f"Mean RMSLE: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"OOF RMSLE:  {oof_score:.5f}")

    oof = pd.DataFrame({
        'Id': train_ids,
        'SalePrice': np.expm1(y_train),
        'OOF_Prediction': np.expm1(oof_predictions),
        'LogSalePrice': y_train,
        'OOF_LogPrediction': oof_predictions,
    })
    oof.to_csv(CV_OOF_PATH, index=False, float_format='%.6f')
    print(f"OOF 预测已保存至: {CV_OOF_PATH}")

    return oof_score, fold_scores, oof_predictions


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
    args = parse_args()
    print("加载数据...")
    train, test = load_data()

    print("数据预处理...")
    all_data, y_train, train, test = preprocess_data(train, test)

    print("特征工程...")
    if args.list_features:
        all_data, feature_names = feature_engineering(all_data, return_feature_names=True)
        write_selected_features(feature_names)
        return
    all_data = feature_engineering(all_data)

    # 分割训练/测试
    X_train = all_data[:len(train)].astype(np.float64)
    X_test = all_data[len(train):].astype(np.float64)

    if args.cv or args.cv_only:
        run_local_cv(
            X_train,
            y_train,
            train['Id'].values,
            n_splits=args.cv_folds,
            n_features=args.n_features
        )
        if args.cv_only:
            return

    # 互信息特征选择（保留200个特征）
    print(f"互信息特征选择（k={args.n_features}）...")
    X_train, X_test, _ = select_top_features(X_train, y_train, X_test, args.n_features)
    print(f"最终特征数: {X_train.shape[1]}")

    # 多种子 stacking + 强单模型 blending
    stacking_pred_log = fit_predict_stacking_ensemble(X_train, y_train, X_test)
    xgb_pred_log, catboost_pred_log = fit_predict_single_model_blend(X_train, y_train, X_test)
    total_weight = args.stacking_weight + args.xgb_weight + args.catboost_weight
    if not np.isclose(total_weight, 1.0):
        print(f"融合权重总和为 {total_weight:.4f}，自动归一化到 1.0")
    stacking_weight = args.stacking_weight / total_weight
    xgb_weight = args.xgb_weight / total_weight
    catboost_weight = args.catboost_weight / total_weight
    print(
        "最终融合权重: "
        f"stacking={stacking_weight:.3f}, "
        f"xgboost={xgb_weight:.3f}, "
        f"catboost={catboost_weight:.3f}"
    )
    final_pred_log = (
        stacking_weight * stacking_pred_log +
        xgb_weight * xgb_pred_log +
        catboost_weight * catboost_pred_log
    )
    final_pred_log = np.clip(final_pred_log, 9.0, 14.0)
    submission = pd.DataFrame({'Id': test['Id'].values, 'SalePrice': np.expm1(final_pred_log)})
    submission.to_csv(SUBMISSION_PATH, index=False, float_format='%.6f')
    print(f"\n✅ submission_optimized.csv 已保存至: {SUBMISSION_PATH}")
    print(f"预测价格范围: ${submission['SalePrice'].min():,.0f} ~ ${submission['SalePrice'].max():,.0f}")
    print(f"预测价格均值: ${submission['SalePrice'].mean():,.0f}")


if __name__ == '__main__':
    main()
