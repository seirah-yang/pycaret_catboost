import warnings
warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd

from scipy.stats import shapiro, kstest, levene, kruskal
import pingouin as pg
import scikit_posthocs as sp
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from pycaret.classification import setup, compare_models, tune_model, blend_models, finalize_model, predict_model, pull


# 1. Data load
train_df = pd.read_csv('/home/alpaco/homework/train.csv')
test_df = pd.read_csv('/home/alpaco/homework/test.csv')

print(train_df.info(), test_df.info())
print(train_df.head(), test_df.head())


# 2. 전처리 / Label Encoding
def encode_categoricals(train_df, test_df, cat_cols=['gender', 'subscription_type']):
    encoders = {}
    for col in cat_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            if col in test_df.columns:
                test_df[col] = test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            encoders[col] = le
    return train_df, test_df, encoders


print(train_df)
print(test_df)

encoders = encode_categoricals(train_df, test_df)


# 3. ANOVA
def select_numeric_features(df, label_cols=('ID', 'support_needs')):
    return [c for c in df.columns if df[c].dtype != 'O' and c not in label_cols]


def ks_normal_test(X):
    X = np.asarray(X)
    X = X[~np.isnan(X)]
    if len(X) < 3:
        return np.nan
    mu, sigma = np.mean(X), np.std(X, ddof=1)
    if sigma <= 0:
        return 1.0
    return kstest(X, 'norm', args=(mu, sigma)).pvalue


def shapiro_or_ks(X, n_threshold=50):
    X = np.asarray(X)
    X = X[~np.isnan(X)]
    if len(X) < 3:
        return np.nan
    if len(X) < n_threshold:
        return shapiro(X).pvalue
    return ks_normal_test(X)


def check_assumptions(df, group_col, feature_cols, alpha=0.05):
    rows = []
    groups = df[group_col].dropna().unique()
    for feat in feature_cols:
        normal_ok = True
        shapiro_ps = {}
        arrays = []

        for g in groups:
            vals = df.loc[df[group_col] == g, feat].values
            arrays.append(vals)
            p = shapiro_or_ks(vals)
            shapiro_ps[g] = p
            if not np.isnan(p) and p < alpha:
                normal_ok = False

        valid_arrays = [np.asarray(a)[~np.isnan(a)] for a in arrays if len(a) >= 2]
        lev_p = levene(*valid_arrays, center="median").pvalue if len(valid_arrays) >= 2 else np.nan
        equal_var = (lev_p >= alpha) if not np.isnan(lev_p) else False

        if normal_ok and equal_var:
            test = "ANOVA"
        elif normal_ok and not equal_var:
            test = "Welch_ANOVA"
        else:
            test = "Kruskal_Wallis"

        rows.append({
            "feature": feat,
            "normal_all_groups": normal_ok,
            "levene_p": lev_p,
            "equal_var": equal_var,
            "recommended_test": test,
            "shapiro_or_ks_p_by_group": shapiro_ps
        })

    return pd.DataFrame(rows)


def run_anova_like(df, group_col, feature, decision_row):
    test = decision_row["recommended_test"]
    if test == "ANOVA":
        res = pg.anova(data=df, dv=feature, between=group_col, detailed=True)
        method = "One-way ANOVA (equal var, normal)"
    elif test == "Welch_ANOVA":
        res = pg.welch_anova(data=df, dv=feature, between=group_col)
        method = "Welch ANOVA (unequal var, normal)"
    else:
        arrays = [df.loc[df[group_col] == g, feature].values for g in df[group_col].unique()]
        kw = kruskal(*arrays)
        res = pd.DataFrame({"method": ["Kruskal-Wallis"], "H": [kw.statistic], "p-unc": [kw.pvalue]})
        method = "Kruskal–Wallis (nonparametric)"
    return method, res


def run_posthoc(df, group_col, feature, decision_row):
    normal_ok = decision_row["normal_all_groups"]
    equal_var = decision_row["equal_var"]
    if normal_ok and equal_var:
        ph = pg.pairwise_tukey(data=df, dv=feature, between=group_col)
        return "Tukey HSD", ph
    elif normal_ok and not equal_var:
        ph = pg.pairwise_gameshowell(data=df, dv=feature, between=group_col)
        return "Games–Howell", ph
    else:
        ph = sp.posthoc_dunn(df, val_col=feature, group_col=group_col, p_adjust="bonferroni")
        return "Dunn (Bonferroni adj.)", ph


def multinomial_logit(train_feat, label_col="support_needs", features=None, standardize=True):
    if features is None:
        features = select_numeric_features(train_feat)

    X_train = train_feat[features].copy()
    y_train = train_feat[label_col].astype(int)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)

    X_train = sm.add_constant(X_train)
    model = sm.MNLogit(y_train, X_train)
    res = model.fit(method="newton", maxiter=200, disp=False)

    print(res.summary())
    return res


# 5. PyCaret AutoML 
def run_pycaret(train_df):
    clf_setup = setup(
        data=train_df,
        target="support_needs",
        session_id=42,
        normalize=True,
        fix_imbalance=True,
        verbose=False
    )

    top3 = compare_models(n_select=3)
    tuned_top3 = [tune_model(m, optimize="AUC") for m in top3]
    blender = blend_models(estimator_list=tuned_top3, fold=5)
    final_model = finalize_model(blender)

print(final_model)


# 6. 실행 
def run_full_pipeline(train_df, test_df):
    features = select_numeric_features(train_df)
    assump = check_assumptions(train_df, group_col="support_needs", feature_cols=features)

    print(assump[["feature", "recommended_test"]])

    # Logistic Regression
    logit_res = multinomial_logit(train_df, label_col="support_needs", features=features)

    # AutoML
    final_model = run_pycaret(train_df)

    # Prediction
    preds = predict_model(final_model, data=test_df)
    submission = preds[['ID', 'prediction_label']].rename(columns={'prediction_label': 'predicted_support_needs'})
    submission.to_csv('submission.csv', index=False)


print(assump, logit_res, final_model)


# abelEncoder 적용
# gender, subscription_type 숫자형 변환
# 통계 전처리: 정규성(Shapiro/KS) + 등분산성(Levene)
# ANOVA / Welch / Kruskal
# Tukey / Games-Howell / Dunn 자동 선택
# Multinomial Logistic Regression
# PyCaret AutoML : 모델 선택 → 튜닝 → 블렌딩 → 예측
# submission 저장
