import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb
from catboost import CatBoostRegressor

# 全局随机种子
SEED = 42
np.random.seed(SEED)

# ---------------------- 1. 加载数据 ----------------------
def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

# ---------------------- 2. 数据预处理 ----------------------
def preprocess_data(train, test):
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice', 'Id'], axis=1, inplace=True)
    
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
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
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
    
    return all_data, train, test

# ---------------------- 3. 特征工程 ----------------------
def feature_engineering(all_data):
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    
    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))
        
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness = skewness[abs(skewness['Skew']) > 0.75]
    
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)
        
    all_data = pd.get_dummies(all_data)
    
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    all_data = selector.fit_transform(all_data)
    
    return all_data

# ---------------------- 4. 模型定义 ----------------------
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=SEED))
elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=SEED))
kernel_ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=SEED)

model_lgb = LGBMRegressor(
    objective='regression',
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=5,
    min_child_samples=15,
    min_split_gain=0.01,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=SEED,
    verbose=-1
)

model_xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=SEED,
    early_stopping_rounds=50
)

model_cb = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.01,
    depth=5,
    verbose=False,
    random_state=SEED,
    early_stopping_rounds=50
)

# ---------------------- 5. Stacking 模型 ----------------------
class StackingAveragedModels(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)

        oof_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)

                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[holdout_idx]
                y_val_fold = y[holdout_idx]

                if isinstance(instance, LGBMRegressor):
                    instance.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
                elif isinstance(instance, XGBRegressor):
                    instance.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
                elif isinstance(instance, CatBoostRegressor):
                    instance.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), use_best_model=True)
                else:
                    instance.fit(X_train_fold, y_train_fold)

                oof_predictions[holdout_idx, i] = instance.predict(X_val_fold)

        self.meta_model_.fit(oof_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

stacked_model = StackingAveragedModels(
    base_models=[lasso, elastic_net, kernel_ridge, gboost, model_xgb, model_cb],
    meta_model=model_lgb
)

# ---------------------- 6. 主流程（无评估，直接跑） ----------------------
def main():
    print("Step 1: 加载数据...")
    train, test = load_data()
    
    print("Step 2: 数据预处理...")
    all_data, train, test = preprocess_data(train, test)
    
    print("Step 3: 特征工程...")
    all_data = feature_engineering(all_data)
    
    X_train = all_data[:len(train)]
    X_test = all_data[len(train):]
    y_train = np.log1p(train['SalePrice']).values
    
    print("Step 4: 模型训练中...（约1分钟）")
    stacked_model.fit(X_train, y_train)
    
    print("Step 5: 生成提交文件...")
    pred = np.expm1(stacked_model.predict(X_test))
    submission = pd.DataFrame({'Id': test['Id'].values, 'SalePrice': pred})
    submission.to_csv('data/submission_improved.csv', index=False)
    print("✅ submission_improved.csv 已保存！")

if __name__ == '__main__':
    main()