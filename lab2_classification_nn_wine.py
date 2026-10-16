
import importlib.util, subprocess, sys

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
            "seaborn": "seaborn", "sklearn": "scikit-learn", "tensorflow": "tensorflow"}
missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Устанавливаются пакеты:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("Все необходимые пакеты уже установлены.")

import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold,
                                     cross_validate, cross_val_predict)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

DATA_FILE = "winequalityN.csv"
TARGET = "good"
QUALITY_THRESHOLD = 7
NUM_COLS = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
            "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
CAT_COLS = ["type"]


def find_file(name):
    for p in (Path(name), Path("data") / name, Path("/content") / name):
        if p.exists():
            return p
    return None


if find_file(DATA_FILE) is None:
    try:
        from google.colab import files
        print("Загрузите файл", DATA_FILE)
        files.upload()
    except ImportError:
        raise FileNotFoundError(f"Положите файл {DATA_FILE} рядом с ноутбуком")

df_raw = pd.read_csv(find_file(DATA_FILE))
print("Размер таблицы:", df_raw.shape)
print("\nТипы столбцов:\n", df_raw.dtypes)
print("\nПропуски по столбцам:\n", df_raw.isna().sum()[df_raw.isna().sum() > 0])
print("\nПолных дубликатов строк:", int(df_raw.duplicated().sum()))
print("\nКоличество вин по оценке качества:\n", df_raw["quality"].value_counts().sort_index())
print("\nТип вина:\n", df_raw["type"].value_counts())
print("Таблица 1. Первые строки исходного датасета Wine Quality")
df_raw.head()

n_before = len(df_raw)
df = df_raw.drop_duplicates().reset_index(drop=True)
df[TARGET] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)
print(f"Удалено дубликатов: {n_before - len(df)}; осталось {len(df)} вин")
print("\nДоли классов (1 - хорошее вино):\n", df[TARGET].value_counts(normalize=True).round(3))
print("Таблица 2. Описательные статистики числовых признаков (после удаления дубликатов)")
df[NUM_COLS].describe().T.round(3)

print("Рисунок 1. Обзор целевой переменной: оценки качества, тип вина и корреляции признаков с good")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
sns.countplot(x="quality", hue=TARGET, data=df, ax=ax[0], palette=["#4C72B0", "#DD8452"])
ax[0].set_title("Распределение оценок качества"); ax[0].legend(title="good")
sns.countplot(x="type", hue=TARGET, data=df, ax=ax[1], palette=["#4C72B0", "#DD8452"])
ax[1].set_title("Тип вина и класс"); ax[1].legend(title="good")
corr_t = df[NUM_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
ax[2].barh(corr_t.index, corr_t.values, color=np.where(corr_t.values > 0, "#55A868", "#C44E52"))
ax[2].set_title("Корреляция признаков с целью (good)")
plt.tight_layout(); plt.show()
print("Рис.1.Признаки распределение, тип вина и класса")
print("Рисунок 2. Boxplot ключевых признаков по классам (0 — обычное, 1 — хорошее вино)")
top = ["alcohol", "density", "volatile acidity", "chlorides", "total sulfur dioxide", "residual sugar"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for a, c in zip(axes.ravel(), top):
    sns.boxplot(x=TARGET, y=c, data=df, ax=a, palette=["#4C72B0", "#DD8452"])
    a.set_title(c); a.set_xlabel("класс (good)")
plt.suptitle("Распределение признаков по классам качества", y=1.02, fontsize=12)
plt.tight_layout(); plt.show()

print("Рисунок 3. Корреляционная матрица числовых признаков и целевой переменной")
plt.figure(figsize=(10, 8))
sns.heatmap(df[NUM_COLS + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, cbar_kws={"shrink": .8})
plt.title("Корреляционная матрица"); plt.tight_layout(); plt.show()

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print("train:", X_train.shape, " test:", X_test.shape)
print("доля класса 1: train=%.3f  test=%.3f" % (y_train.mean(), y_test.mean()))


def make_pre(scaler):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", scaler)])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(drop="if_binary", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


pre_std, pre_mm = make_pre(StandardScaler()), make_pre(MinMaxScaler())
DATA = {
    "std": (pre_std.fit_transform(X_train), pre_std.transform(X_test)),
    "mm":  (pre_mm.fit_transform(X_train),  pre_mm.transform(X_test)),
}
FEATURE_NAMES = [n.split("__", 1)[1] for n in pre_std.get_feature_names_out()]
print("признаков после кодирования:", DATA["std"][0].shape[1], "->", FEATURE_NAMES[-2:])
print("пропусков после предобработки:", int(np.isnan(DATA["std"][0]).sum()))


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics_from(yte, pred, sc):
    return {"Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred),
            "F1": f1_score(yte, pred),
            "AUC-ROC": roc_auc_score(yte, sc)}


RESULTS_BASE, RESULTS_TUNED, RESULTS_TUNED_DEF, ROC_TUNED = {}, {}, {}, {}
BEST_PARAMS, THRESHOLDS, BEST_CV, TEST_SCORES = {}, {}, {}, {}
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)


def baseline(name, model, data_key):
    Xtr, Xte = DATA[data_key]
    model.fit(Xtr, y_train)
    RESULTS_BASE[name] = metrics_from(y_test, model.predict(Xte), get_scores(model, Xte))


def best_threshold(model, Xtr):
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    oof = cross_val_predict(model, Xtr, y_train, cv=CV, method=method)
    oof = oof[:, 1] if oof.ndim == 2 else oof
    cands = np.unique(np.quantile(oof, np.linspace(0.02, 0.98, 97)))
    f1s = [f1_score(y_train, oof >= t) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def param_curve(estimator, data_key, param, values, title="", log_x=False):
    Xtr = DATA[data_key][0]
    rows = []
    for v in values:
        r = cross_validate(estimator.set_params(**{param: v}), Xtr, y_train, cv=CV,
                           scoring={"auc": "roc_auc", "f1": "f1"}, return_train_score=True)
        rows.append({param: v, "AUC_train": r["train_auc"].mean(), "AUC_cv": r["test_auc"].mean(),
                     "F1_train": r["train_f1"].mean(), "F1_cv": r["test_f1"].mean()})
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    xs = res[param].astype(str) if not pd.api.types.is_numeric_dtype(res[param]) else res[param]
    for a, (c1, c2, t) in zip(ax, [("AUC_train", "AUC_cv", "AUC-ROC"), ("F1_train", "F1_cv", "F1-score")]):
        a.plot(xs, res[c1], "o-", label="train")
        a.plot(xs, res[c2], "s-", label="кросс-валидация")
        a.set_xlabel(param); a.set_title(f"{title} {t}"); a.legend()
        if log_x:
            a.set_xscale("log")
    fig.suptitle(f"Рисунок. Влияние параметра «{param}» ({title})" if title else f"Рисунок. Влияние параметра «{param}»", y=1.02, fontsize=11)
    plt.tight_layout(); plt.show()
    return res


def run_grid(name, estimator, grid, data_key):
    Xtr, Xte = DATA[data_key]
    gs = GridSearchCV(estimator, grid, scoring={"auc": "roc_auc", "f1": "f1"}, refit="auc",
                      cv=CV, n_jobs=-1).fit(Xtr, y_train)
    best = gs.best_estimator_
    sc = get_scores(best, Xte)
    thr = best_threshold(best, Xtr)
    RESULTS_TUNED_DEF[name] = metrics_from(y_test, best.predict(Xte), sc)
    RESULTS_TUNED[name] = metrics_from(y_test, sc >= thr, sc)
    BEST_PARAMS[name], THRESHOLDS[name], BEST_CV[name] = gs.best_params_, thr, gs.best_score_
    TEST_SCORES[name] = sc
    ROC_TUNED[name] = roc_curve(y_test, sc)[:2] + (RESULTS_TUNED[name]["AUC-ROC"],)
    print(f"{name}: лучшие параметры {gs.best_params_}, CV AUC = {gs.best_score_:.4f}, порог = {thr:.3f}")
    cvr = pd.DataFrame(gs.cv_results_)

    params = pd.DataFrame([{k: ("None" if v is None else v) for k, v in p.items()}
                           for p in cvr["params"]]).fillna("-")
    out = pd.concat([params, cvr[["mean_test_auc", "mean_test_f1"]]
                     .rename(columns={"mean_test_auc": "CV AUC", "mean_test_f1": "CV F1"})], axis=1)
    return out.sort_values("CV AUC", ascending=False).round(4).reset_index(drop=True)


baseline("GaussianNB",   GaussianNB(),                              "std")
baseline("MultinomialNB", MultinomialNB(),                          "mm")
baseline("ComplementNB", ComplementNB(),                            "mm")
baseline("BernoulliNB",  BernoulliNB(),                             "std")
baseline("DecisionTree", DecisionTreeClassifier(random_state=SEED), "std")
baseline("LDA",          LinearDiscriminantAnalysis(),              "std")
baseline("SVM",          SVC(random_state=SEED),                    "std")
baseline("kNN",          KNeighborsClassifier(),                    "std")
print("Таблица 3. Метрики baseline-моделей на тестовой выборке (порог 0.5)")
pd.DataFrame(RESULTS_BASE).T.round(4)

r = param_curve(GaussianNB(), "std", "var_smoothing", np.logspace(-11, 2, 14), "GaussianNB", log_x=True)
print("Таблица 4. GridSearchCV: GaussianNB (топ по CV AUC)")
run_grid("GaussianNB", GaussianNB(), {"var_smoothing": np.logspace(-11, 2, 14)}, "std").head()

alphas = [1e-3, 1e-2, 0.1, 0.5, 1, 2, 5, 10, 30, 100]
r = param_curve(MultinomialNB(), "mm", "alpha", alphas, "MultinomialNB", log_x=True)
print("Таблица 5. GridSearchCV: MultinomialNB (топ по CV AUC)")
run_grid("MultinomialNB", MultinomialNB(), {"alpha": alphas}, "mm").head()

r = param_curve(ComplementNB(), "mm", "alpha", alphas, "ComplementNB", log_x=True)
print("Таблица 6. GridSearchCV: ComplementNB (топ по CV AUC)")
run_grid("ComplementNB", ComplementNB(), {"alpha": alphas, "norm": [False, True]}, "mm").head()

r = param_curve(BernoulliNB(), "std", "binarize", [0.0, 0.25, 0.5, 1.0, 1.5], "BernoulliNB")
print("Таблица 7. GridSearchCV: BernoulliNB (топ по CV AUC)")
run_grid("BernoulliNB", BernoulliNB(), {"binarize": [0.0, 0.25, 0.5, 1.0], "alpha": [0.1, 1, 5, 20]}, "std").head()

r = param_curve(DecisionTreeClassifier(random_state=SEED), "std", "max_depth",
                [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, None], "DecisionTree")
r = param_curve(DecisionTreeClassifier(random_state=SEED), "std", "min_samples_leaf",
                [1, 2, 5, 10, 25, 50, 100], "DecisionTree", log_x=True)
print("Таблица 8. GridSearchCV: DecisionTree (топ по CV AUC)")
run_grid("DecisionTree", DecisionTreeClassifier(random_state=SEED),
         {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [5, 10, 25, 50, 100],
          "criterion": ["gini", "entropy"], "class_weight": [None, "balanced"]}, "std").head(8)

r = param_curve(LinearDiscriminantAnalysis(solver="lsqr"), "std", "shrinkage", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], "LDA")
print("Таблица 9. GridSearchCV: LDA (топ по CV AUC)")
run_grid("LDA", LinearDiscriminantAnalysis(),
         [{"solver": ["svd"]}, {"solver": ["lsqr", "eigen"], "shrinkage": [None, "auto", 0.1, 0.5, 0.9]}], "std").head(8)

r = param_curve(SVC(kernel="rbf", random_state=SEED), "std", "C", [0.01, 0.1, 1, 10, 100], "SVM (rbf)", log_x=True)
r = param_curve(SVC(kernel="rbf", C=1, random_state=SEED), "std", "gamma", [1e-4, 1e-3, 1e-2, 0.1, 1], "SVM (rbf)", log_x=True)
r = param_curve(SVC(C=1, random_state=SEED), "std", "kernel", ["linear", "poly", "rbf", "sigmoid"], "SVM")
print("Таблица 10. GridSearchCV: SVM (топ по CV AUC)")
run_grid("SVM", SVC(random_state=SEED),
         {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1], "kernel": ["rbf"],
          "class_weight": [None, "balanced"]}, "std").head(8)

r = param_curve(KNeighborsClassifier(), "std", "n_neighbors", [1, 3, 5, 9, 15, 25, 51, 101, 201], "kNN", log_x=True)
r = param_curve(KNeighborsClassifier(n_neighbors=15), "std", "p", [1, 2, 3], "kNN (метрика Минковского)")
print("Таблица 11. GridSearchCV: kNN (топ по CV AUC)")
run_grid("kNN", KNeighborsClassifier(),
         {"n_neighbors": [5, 15, 25, 51, 101, 151], "weights": ["uniform", "distance"], "p": [1, 2]}, "std").head(8)

cmp = pd.DataFrame({
    "F1 baseline": pd.DataFrame(RESULTS_BASE).T["F1"],
    "F1 tuned, порог по умолч.": pd.DataFrame(RESULTS_TUNED_DEF).T["F1"],
    "F1 tuned, порог подобран": pd.DataFrame(RESULTS_TUNED).T["F1"],
    "AUC baseline": pd.DataFrame(RESULTS_BASE).T["AUC-ROC"],
    "AUC tuned": pd.DataFrame(RESULTS_TUNED).T["AUC-ROC"],
    "порог": pd.Series(THRESHOLDS),
}).round(4)
print("Таблица 12. Сравнение F1 и AUC-ROC: baseline vs настройка vs подбор порога")
cmp

print("Рисунок 4. Сравнение AUC-ROC и F1 на тесте: baseline vs после настройки гиперпараметров")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
order = pd.DataFrame(RESULTS_TUNED).T["AUC-ROC"].sort_values().index
y_pos = np.arange(len(order))
ax[0].barh(y_pos - 0.2, pd.DataFrame(RESULTS_BASE).T.loc[order, "AUC-ROC"], 0.4, label="baseline")
ax[0].barh(y_pos + 0.2, pd.DataFrame(RESULTS_TUNED).T.loc[order, "AUC-ROC"], 0.4, label="после настройки")
ax[0].set_yticks(y_pos); ax[0].set_yticklabels(order); ax[0].set_xlim(0.5, 0.9)
ax[0].set_title("AUC-ROC на тесте: до и после настройки"); ax[0].legend()
ax[1].barh(y_pos - 0.2, pd.DataFrame(RESULTS_BASE).T.loc[order, "F1"], 0.4, label="baseline")
ax[1].barh(y_pos + 0.2, pd.DataFrame(RESULTS_TUNED).T.loc[order, "F1"], 0.4, label="настройка + порог")
ax[1].set_yticks(y_pos); ax[1].set_yticklabels(order); ax[1].set_xlim(0, 0.7)
ax[1].set_title("F1 на тесте: до и после настройки"); ax[1].legend()
plt.tight_layout(); plt.show()

import datetime, shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["PYTHONHASHSEED"] = str(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

Xtrain_all, Xte_nn = DATA["std"]
Xtr_nn, Xval_nn, ytr_nn, yval_nn = train_test_split(
    Xtrain_all, y_train, test_size=0.2, stratify=y_train, random_state=SEED)
ytr_nn, yval_nn = ytr_nn.astype("float32"), yval_nn.astype("float32")
print("обучение:", Xtr_nn.shape, " валидация:", Xval_nn.shape, " тест:", Xte_nn.shape)

LOG_ROOT = Path("logs")
shutil.rmtree(LOG_ROOT, ignore_errors=True)


def build_model(n_layers=2, units=64, activation="relu", lr=1e-3, dropout=0.2):
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([layers.Input(shape=(Xtr_nn.shape[1],))])
    for i in range(n_layers):
        m.add(layers.Dense(max(units // (2 ** i), 8), activation=activation))
        if dropout:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m


def pick_threshold(y_true, prob):
    cands = np.unique(np.quantile(prob, np.linspace(0.02, 0.98, 97)))
    f1s = [f1_score(y_true, prob >= t) for t in cands]
    return float(cands[int(np.argmax(f1s))])


NN_RES, NN_HIST = {}, {}


def run_nn(name, epochs=60, batch_size=32, patience=8, class_weight=None, **kw):
    m = build_model(**kw)
    cbs = [keras.callbacks.TensorBoard(log_dir=str(LOG_ROOT / name))]
    if patience is not None:
        cbs.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True))
    t0 = time.time()
    h = m.fit(Xtr_nn, ytr_nn, validation_data=(Xval_nn, yval_nn),
              epochs=epochs, batch_size=batch_size, class_weight=class_weight,
              callbacks=cbs, verbose=0)
    val_prob = m.predict(Xval_nn, verbose=0).ravel()
    prob = m.predict(Xte_nn, verbose=0).ravel()
    thr = pick_threshold(yval_nn, val_prob)
    res = metrics_from(y_test, prob >= thr, prob)
    NN_RES[name] = {"val AUC": roc_auc_score(yval_nn, val_prob), **res,
                    "порог": thr, "эпох": len(h.history["loss"]), "время, с": time.time() - t0}
    NN_HIST[name] = (h.history, prob)
    return m


def plot_hist(names, caption=None):
    if caption:
        print(caption)
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
    for n in names:
        h = NN_HIST[n][0]
        for a, (tr, va) in zip(ax, [("loss", "val_loss"), ("accuracy", "val_accuracy"), ("auc", "val_auc")]):
            line, = a.plot(h[va], label=n)
            a.plot(h[tr], "--", color=line.get_color(), alpha=.5)
    for a, t in zip(ax, ["Loss", "Accuracy", "AUC"]):
        a.set_title(f"{t} (сплошная — валидация, пунктир — обучение)", fontsize=9)
        a.set_xlabel("эпоха"); a.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def nn_table(names):
    return pd.DataFrame({n: NN_RES[n] for n in names}).T.round(4)


base_model = run_nn("base")
base_model.summary()
plot_hist(["base"], caption="Рисунок 5. Кривые обучения базовой нейронной сети")
print("Таблица 13. Базовая нейронная сеть: метрики на валидации и тесте")
nn_table(["base"])

batches = [16, 32, 64, 128, 256]
for bs in batches:
    if bs != 32:
        run_nn(f"batch_{bs}", batch_size=bs)
NN_HIST["batch_32"], NN_RES["batch_32"] = NN_HIST["base"], NN_RES["base"]
plot_hist([f"batch_{b}" for b in batches], caption="Рисунок 6. Кривые обучения: влияние размера батча")
print("Таблица 14. Нейросеть: влияние размера батча")
nn_table([f"batch_{b}" for b in batches])

epoch_list = [5, 20, 60, 120]
for ep in epoch_list:
    run_nn(f"epochs_{ep}", epochs=ep, patience=None)
plot_hist([f"epochs_{e}" for e in epoch_list], caption="Рисунок 7. Кривые обучения: влияние числа эпох")
print("Таблица 15. Нейросеть: влияние числа эпох (без ранней остановки)")
nn_table([f"epochs_{e}" for e in epoch_list])

archs = [(1, 16), (1, 64), (2, 64), (3, 64), (3, 128)]
for L, U in archs:
    if (L, U) != (2, 64):
        run_nn(f"L{L}_U{U}", n_layers=L, units=U)
NN_HIST["L2_U64"], NN_RES["L2_U64"] = NN_HIST["base"], NN_RES["base"]
plot_hist([f"L{L}_U{U}" for L, U in archs], caption="Рисунок 8. Кривые обучения: влияние архитектуры сети")
print("Таблица 16. Нейросеть: влияние архитектуры (число слоёв и нейронов)")
nn_table([f"L{L}_U{U}" for L, U in archs])

acts = ["relu", "tanh", "sigmoid", "elu", "linear"]
for a in acts:
    if a != "relu":
        run_nn(f"act_{a}", activation=a)
NN_HIST["act_relu"], NN_RES["act_relu"] = NN_HIST["base"], NN_RES["base"]
plot_hist([f"act_{a}" for a in acts], caption="Рисунок 9. Кривые обучения: влияние функции активации")
print("Таблица 17. Нейросеть: влияние функции активации")
nn_table([f"act_{a}" for a in acts])

for lr in [1e-4, 1e-2]:
    run_nn(f"lr_{lr}", lr=lr)
NN_HIST["lr_0.001"], NN_RES["lr_0.001"] = NN_HIST["base"], NN_RES["base"]
for dr in [0.0, 0.5]:
    run_nn(f"drop_{dr}", dropout=dr)
NN_HIST["drop_0.2"], NN_RES["drop_0.2"] = NN_HIST["base"], NN_RES["base"]
w1 = (1 - y_train.mean()) / y_train.mean()
run_nn("class_weight", class_weight={0: 1.0, 1: float(w1)})
plot_hist(["lr_0.0001", "lr_0.001", "lr_0.01"], caption="Рисунок 10. Кривые обучения: влияние скорости обучения")
plot_hist(["drop_0.0", "drop_0.2", "drop_0.5", "class_weight"], caption="Рисунок 11. Кривые обучения: Dropout и веса классов")
print("Таблица 18. Нейросеть: скорость обучения, Dropout и веса классов")
nn_table(["lr_0.0001", "lr_0.001", "lr_0.01", "drop_0.0", "drop_0.2", "drop_0.5", "class_weight"])

groups = {
    "размер батча":   [f"batch_{b}" for b in batches],
    "число эпох":     [f"epochs_{e}" for e in epoch_list],
    "архитектура":    [f"L{L}_U{U}" for L, U in archs],
    "активация":      [f"act_{a}" for a in acts],
    "скорость обуч.": ["lr_0.0001", "lr_0.001", "lr_0.01"],
    "dropout":        ["drop_0.0", "drop_0.2", "drop_0.5"],
}
for title, names in groups.items():
    t = pd.DataFrame({n: NN_RES[n] for n in names}).T
    b, w = t["val AUC"].idxmax(), t["val AUC"].idxmin()
    print(f"{title:>15}: лучший {b} (val AUC {t.loc[b, 'val AUC']:.4f}, test F1 {t.loc[b, 'F1']:.4f}, эпох {int(t.loc[b, 'эпох'])}); "
          f"худший {w} (val AUC {t.loc[w, 'val AUC']:.4f}); разброс val AUC = {t['val AUC'].max() - t['val AUC'].min():.4f}")

all_nn = pd.DataFrame(NN_RES).T.drop_duplicates(subset=["val AUC", "эпох"])
best_nn = all_nn["val AUC"].astype(float).idxmax()
print("Лучшая конфигурация нейросети по val AUC:", best_nn)
RESULTS_TUNED["NeuralNet"] = {k: NN_RES[best_nn][k] for k in ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]}
fpr, tpr, _ = roc_curve(y_test, NN_HIST[best_nn][1])
ROC_TUNED["NeuralNet"] = (fpr, tpr, NN_RES[best_nn]["AUC-ROC"])
TEST_SCORES["NeuralNet"] = NN_HIST[best_nn][1]
print("Таблица 19. Топ-8 конфигураций нейросети по валидационному AUC-ROC")
all_nn.sort_values("val AUC", ascending=False).head(8).round(4)

final = pd.DataFrame(RESULTS_TUNED).T.round(4).sort_values("AUC-ROC", ascending=False)
final.to_csv("final_metrics.csv")
print("Таблица 20. Итоговые метрики всех настроенных моделей на тестовой выборке")
final

print("Рисунок 12. Сравнение моделей по метрикам на тестовой выборке")
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
fig, axes = plt.subplots(2, 3, figsize=(17, 8)); axes = axes.ravel()
for a, mt in zip(axes, metrics):
    s = final[mt].sort_values()
    a.barh(s.index, s.values, color=sns.color_palette("viridis", len(s)))
    for i, v in enumerate(s.values):
        a.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    a.set_title(mt); a.set_xlim(0, 1.05)
axes[-1].axis("off")
plt.tight_layout(); plt.show()

print("Рисунок 13. ROC-кривые настроенных моделей на тестовой выборке")
plt.figure(figsize=(7.5, 6))
for n, (f, t, auc) in ROC_TUNED.items():
    plt.plot(f, t, label=f"{n} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="случайный классификатор")
plt.xlabel("FPR (доля ложных тревог)"); plt.ylabel("TPR (Recall)")
plt.title("ROC-кривые настроенных моделей"); plt.legend(fontsize=8); plt.show()

rng = np.random.RandomState(SEED)
boot_idx = [rng.randint(0, len(y_test), len(y_test)) for _ in range(500)]
rows = {}
for name, sc in TEST_SCORES.items():
    aucs = [roc_auc_score(y_test[i], sc[i]) for i in boot_idx]
    rows[name] = {"AUC-ROC": roc_auc_score(y_test, sc),
                  "2.5%": np.percentile(aucs, 2.5), "97.5%": np.percentile(aucs, 97.5)}
ci = pd.DataFrame(rows).T.sort_values("AUC-ROC", ascending=False)

print("Рисунок 14. 95%-е доверительные интервалы AUC-ROC на тесте (bootstrap)")
plt.figure(figsize=(8, 4.2))
yy = np.arange(len(ci))[::-1]
plt.errorbar(ci["AUC-ROC"], yy, xerr=[ci["AUC-ROC"] - ci["2.5%"], ci["97.5%"] - ci["AUC-ROC"]],
             fmt="o", capsize=4)
plt.yticks(yy, ci.index); plt.xlabel("AUC-ROC (95% доверительный интервал)")
plt.title("Различимы ли модели по AUC на тесте?"); plt.tight_layout(); plt.show()
print("Таблица 21. 95%-е доверительные интервалы AUC-ROC на тесте (bootstrap, 500 повторов)")
ci.round(4)

best_auc, best_f1 = final["AUC-ROC"].idxmax(), final["F1"].idxmax()
print(f"Наибольший AUC-ROC: {best_auc} ({final.loc[best_auc, 'AUC-ROC']:.4f})")
print(f"Наибольший F1:      {best_f1} ({final.loc[best_f1, 'F1']:.4f})")
print("Модели, чей 95% интервал AUC пересекается с интервалом лучшей:",
      [n for n in ci.index if ci.loc[n, "97.5%"] >= ci.loc[best_auc, "2.5%"] and ci.loc[best_auc, "97.5%"] >= ci.loc[n, "2.5%"] and n != best_auc])
if "NeuralNet" in final.index:
    rank = list(final.index).index("NeuralNet") + 1
    print(f"Нейросеть: AUC = {final.loc['NeuralNet', 'AUC-ROC']:.4f}, F1 = {final.loc['NeuralNet', 'F1']:.4f}, место по AUC: {rank} из {len(final)}")
else:
    print("Нейросеть в сравнение не включена: сначала выполните раздел 6.")

final = pd.DataFrame(RESULTS_TUNED).T.sort_values("AUC-ROC", ascending=False)
if "NeuralNet" not in final.index:
    print("Сначала выполните раздел 6 (нейронная сеть), затем перезапустите ячейки раздела 7.")
else:
    rng = np.random.RandomState(SEED)
    boot_idx = [rng.randint(0, len(y_test), len(y_test)) for _ in range(500)]
    ci = pd.DataFrame({n: {"AUC": roc_auc_score(y_test, sc),
                           "lo": np.percentile([roc_auc_score(y_test[i], sc[i]) for i in boot_idx], 2.5),
                           "hi": np.percentile([roc_auc_score(y_test[i], sc[i]) for i in boot_idx], 97.5)}
                       for n, sc in TEST_SCORES.items()}).T
    best_cl = final.drop(index="NeuralNet")["AUC-ROC"].idxmax()
    nn, bc = final.loc["NeuralNet"], final.loc[best_cl]
    overlap = ci.loc["NeuralNet", "lo"] <= ci.loc[best_cl, "hi"] and ci.loc[best_cl, "lo"] <= ci.loc["NeuralNet", "hi"]
    print(f"Лучшая конфигурация нейросети (по val AUC): {best_nn}")
    print(f"Нейросеть:  AUC-ROC = {nn['AUC-ROC']:.4f}, F1 = {nn['F1']:.4f}, Recall = {nn['Recall']:.4f}, Precision = {nn['Precision']:.4f}")
    print(f"{best_cl}: AUC-ROC = {bc['AUC-ROC']:.4f}, F1 = {bc['F1']:.4f} (лучший классический метод по AUC)")
    print(f"Разность AUC (нейросеть - {best_cl}) = {nn['AUC-ROC'] - bc['AUC-ROC']:+.4f}")
    print("95% интервалы AUC", "пересекаются: различие статистически не значимо." if overlap
          else "не пересекаются: различие статистически значимо.")
    place = list(final.index).index("NeuralNet") + 1
    print(f"Место нейросети по AUC-ROC: {place} из {len(final)}.")