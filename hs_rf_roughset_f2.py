import cudf
import cupy as cp
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import f1_score, recall_score
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.model_selection import StratifiedKFold
import random
import os
from roughset import *
from type2fuzzy import fuzzy2_preprocess

#==============COLOURS========================================
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"
CYAN = "\033[36m"
DIM = "\033[2m"
BG_BRIGHT_GREEN = "\033[102m"

#=================Hyperparameters===============================
NI = 50
BW = 0.6
seeds = [42, 7, 61]
results = []

dataset = "anemia"
file_path = f'/mnt/c/Users/ralin/Documents/RF_HS_FF/data/{dataset}.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

def addNoise(X, noise):
    if noise > 0:
        print(f"noise={noise}")
        for col in X.columns:
            std = X[col].std()
            noise_x = cp.random.normal(0, noise * std, len(X))
            X[col] += noise_x

for RANDOM_SEED in seeds:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    cp.random.seed(RANDOM_SEED)
    
    df = cudf.read_csv(file_path)
    
    _class = "Diagnosis"
    
    unique_diagnoses = df[_class].unique()
    diagnosis_map = {diag: i for i, diag in enumerate(unique_diagnoses.to_numpy())}
    df[_class] = df[_class].map(diagnosis_map)
    print("Mapped Diagnosis values:", df[_class].unique().to_numpy())
    
    print("Shuffling dataset...")
    df = df.sample(frac=1.0, random_state=RANDOM_SEED)
    
    y = df[_class]
    X = df.drop(_class, axis=1)

    df_cpu = df.to_pandas()
    X_cpu = X.to_pandas()
    y_cpu = y.to_pandas()

    # features = greedy_reduct(df_cpu, X_cpu.columns.tolist(), _class, max_features=20)
    # X = X[features] # for reduct

    # Data Leakage in StratifiedKFold
    # done, now check  - Fuzzy expansion inside the CV loop, only on the training fold, then apply the same transformation to the validation fold. This way the validation data won't influence the fuzzy sets.
    #X = fuzzy2_preprocess(X, y, n_terms=2, verbose=True)
    

    noise = 0.0
    addNoise(X, noise)
   
    
    means = X.mean().to_pandas().round(3)
    stds = X.std().to_pandas().round(3)
    print(f"mean: {means},\n std: {stds}")
    
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = list(skf.split(X_np, y_np))
    
    X_gpu = X
    y_gpu = y
    
    maxf1mean = 0
    it_update = 0
    
    def rf_fitness(harmony, ite):
        global it_update, maxf1mean
        n_estimators = int(harmony[0])
        max_depth = int(harmony[1])
        max_features = float(harmony[2])
        
        f1s = []
        clf = cuRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=RANDOM_SEED,
             #   min_samples_leaf=2
            )
        for train_idx, val_idx in cv_splits:
            X_train = X_gpu.iloc[train_idx]
            X_val = X_gpu.iloc[val_idx]
            y_train = y_gpu.iloc[train_idx]
            y_val = y_gpu.iloc[val_idx]
            
#   ====================== Fuzzy Type-2 
            X_train_fuzzy = fuzzy2_preprocess(X_train, n_terms=2, verbose=False)
            X_val_fuzzy = fuzzy2_preprocess(X_val, n_terms=2, verbose=False)
            clf.fit(X_train_fuzzy, y_train)
            preds = clf.predict(X_val_fuzzy)
            
            # clf.fit(X_train, y_train)
            # preds = clf.predict(X_val)
            
            y_val_cpu = y_val.to_numpy()
            preds_cpu = preds.get() if hasattr(preds, 'get') else cp.asnumpy(preds)
            
            f1 = f1_score(y_val_cpu, preds_cpu, average='macro')
            f1s.append(float(f1))
        
        currentf1mean = np.mean(f1s)
        if currentf1mean > maxf1mean:
            maxf1mean = currentf1mean
            it_update = ite
            print(f"{GREEN}{ite}: Estimators={n_estimators}, Depth={max_depth}, Features={max_features:.2f} → F1={currentf1mean:.4f} <-----------------------------------------------------------------------------{RESET}")
        else:
            print(f"{YELLOW}{ite}{RESET}, {CYAN}{maxf1mean:.4f}{RESET}: {DIM}Estimators={n_estimators}, Depth={max_depth}, Features={max_features:.2f} →{RESET} F1={currentf1mean:.4f}")
        
        return 1 - currentf1mean
    
    HM = []
    
    def harmony_search(obj_func, var_bounds, hms, iterations):
        for _ in range(hms):
            harmony = [
                random.randint(lb, ub) if t == 'int' else random.uniform(lb, ub)
                for lb, ub, t in var_bounds
            ]
            fitness = obj_func(harmony, -1)
            HM.append((harmony, fitness))
        
        HM.sort(key=lambda x: x[1])
        best = HM[0]
        
        for ite in range(iterations):
            hmcr = 0.6
            par = 0.8
            
            new_harmony = []
            for i, (lb, ub, t) in enumerate(var_bounds):
                if random.random() < hmcr:
                    value = random.choice(HM)[0][i]
                    if random.random() < par:
                        value += random.uniform(-BW, BW) * (ub - lb)
                else:
                    value = random.randint(lb, ub) if t == 'int' else random.uniform(lb, ub)
                
                value = int(np.clip(round(value), lb, ub)) if t == 'int' else float(np.clip(value, lb, ub))
                new_harmony.append(value)
            
            fitness = obj_func(new_harmony, ite)
            if fitness < HM[-1][1]:
                HM[-1] = (new_harmony, fitness)
                HM.sort(key=lambda x: x[1])
                best = HM[0]
            elif len(HM)<hms:
                print("smaller than 5")
                HM.append((new_harmony, fitness))
                HM.sort(key=lambda x: x[1])
                best = HM[0]
        
        return best[0], best[1]
    
    var_bounds = [
        (10, 100, 'int'),
        (5, 27, 'int'),
        (0.4, 1, 'float')
    ]
    
    best_harmony, best_fitness = harmony_search(
        obj_func=rf_fitness,
        var_bounds=var_bounds,
        hms=5,
        iterations=NI
    )
    
    n, d, f = best_harmony
    final_clf = cuRF(
        n_estimators=int(n),
        max_depth=int(d),
        max_features=float(f),
        random_state=RANDOM_SEED
    )
    
    macro_f1s = []
    weighted_f1s = []
    weighted_recalls = []
    f1s = []
    recalls = []
    
    for train_idx, val_idx in cv_splits:
        X_train = X_gpu.iloc[train_idx]
        X_val = X_gpu.iloc[val_idx]
        y_train = y_gpu.iloc[train_idx]
        y_val = y_gpu.iloc[val_idx]
        
        final_clf.fit(X_train, y_train)
        preds = final_clf.predict(X_val)
        
        y_val_cpu = y_val.to_numpy()
        preds_cpu = preds.get() if hasattr(preds, 'get') else cp.asnumpy(preds)
        
        weighted_f1 = f1_score(y_val_cpu, preds_cpu, average='weighted')
        macro_f1 = f1_score(y_val_cpu, preds_cpu, average='macro')
        weighted_recall = recall_score(y_val_cpu, preds_cpu, average='weighted')
        
        weighted_f1s.append(weighted_f1)
        macro_f1s.append(macro_f1)
        weighted_recalls.append(weighted_recall)
        
        f1_per_class = f1_score(y_val_cpu, preds_cpu, average=None)
        recall_per_class = recall_score(y_val_cpu, preds_cpu, average=None)
        print(f"Per-class F1: {f1_per_class}")
        print(f"Per-class Recall: {recall_per_class}")
        
        if len(f1_per_class) > 1:
            f1s.append(float(f1_per_class[1]))
            recalls.append(float(recall_per_class[1]))
    
    print(f"FINAL → Weighted F1={np.mean(weighted_f1s):.4f}, Macro F1={np.mean(macro_f1s):.4f}, Weighted Recall={np.mean(weighted_recalls):.4f}")
    print(f"Macro F1={np.mean(macro_f1s):.4f}")
    print("\n=== Final cuML Random Forest Evaluation ===")
    print(f"Best Parameters: n_estimators={int(n)}, max_depth={int(d)}, max_features={f:.2f}")
    print(f"Fitness: {1-best_fitness:.4f}")
    print(f"it_update={it_update}")
    
    results.append({
        "randomseed": RANDOM_SEED,
        "n": int(n),
        "d": int(d),
        "max_feat": f,
        "fitness_f1": 1 - best_fitness,
        "macro_f1": np.mean(macro_f1s),
        "weighted_f1": np.mean(weighted_f1s),
        "weighted_recall": np.mean(weighted_recalls),
        "it_update": it_update,
        "noise": noise
    })

import json
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(json.dumps(results, default=float, indent=2))

print(f"\nnoise={noise}")
print(f"name of file: hs_rf_roughset_f2.py")
print(f"Total seeds processed: {len(results)}")
print(f"{BG_BRIGHT_GREEN}with type2fuzzy \n without roughsets{RESET}")

