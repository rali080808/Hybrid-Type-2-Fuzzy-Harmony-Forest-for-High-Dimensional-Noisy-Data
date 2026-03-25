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

dataset = "heart" # "creditcard" or "anemia"
nameoffile = "all_anemia"
seeds = [42, 6, 89, 65, 22]#[ 23, 34, 45, 56, 67]
NI = 200
BW = 0.6

Iterations = ctrl.Antecedent(np.arange(0, 11, 0.1), 'Iterations')
Diversity = ctrl.Antecedent(np.arange(0, 11, 0.1), 'Diversity')
HMCR = ctrl.Consequent(np.arange(0, 11, 1), 'HMCR')
PAR = ctrl.Consequent(np.arange(0, 11, 1), 'PAR')

# Membership functions

for var in [Iterations, Diversity, HMCR, PAR]:
    var['Low'] = fuzz.trimf(var.universe, [0, 0, 3.5])
    var['MediumLow'] = fuzz.trimf(var.universe, [2, 3.5, 5])
    var['Medium'] = fuzz.trimf(var.universe, [3, 5, 7.5])
    var['MediumHigh'] = fuzz.trimf(var.universe, [5, 7.5, 8])
    var['High'] = fuzz.trimf(var.universe, [7, 10, 10])

# Rules 
rules = [
ctrl.Rule(Iterations['Low'] & Diversity['Low'], (HMCR['High'], PAR['Low'])),
ctrl.Rule(Iterations['Low'] & Diversity['Medium'], (HMCR['MediumLow'], PAR['MediumHigh'])),
ctrl.Rule(Iterations['Low'] & Diversity['High'], (HMCR['Medium'], PAR['Medium'])),
ctrl.Rule(Iterations['Medium'] & Diversity['Low'], (HMCR['Medium'], PAR['MediumLow'])),
ctrl.Rule(Iterations['Medium'] & Diversity['Medium'], (HMCR['Medium'], PAR['Medium'])),
ctrl.Rule(Iterations['Medium'] & Diversity['High'], (HMCR['Medium'], PAR['High'])),
ctrl.Rule(Iterations['High'] & Diversity['Low'], (HMCR['Medium'], PAR['High'])),
ctrl.Rule(Iterations['High'] & Diversity['Medium'], (HMCR['MediumHigh'], PAR['MediumLow'])),
ctrl.Rule(Iterations['High'] & Diversity['High'], (HMCR['High'], PAR['High']))
]

hmcr_par_ctrl = ctrl.ControlSystem(rules)
hmcr_par_sim = ctrl.ControlSystemSimulation(hmcr_par_ctrl)
def scale_to_fuzzy_range(value, max_val, scaled_max=10):
    value = np.array(value, dtype=float)
    return value / max_val * scaled_max

def compute_diversity(population: np.ndarray) -> float:
    """Compute population diversity."""
    scaled_population = [list(row) for row in zip(*population[::-1])]
    scaled_population = np.array(scaled_population)

    for i in range(0, len(scaled_population)):
        scaled_population[i] = scale_to_fuzzy_range(scaled_population[i], max(scaled_population[i]))

    mean_vector = np.mean(scaled_population, axis=1)
    #print(f"mean_vector={mean_vector}")
    '''TODO scale the items of the population evenly bc number of trees is much bigger
    than partition of features and hence has a much greater impact'''
    distances = np.zeros(len(scaled_population))
    for i in range(0, len(scaled_population)):
        distances[i] = np.sqrt(np.sum((scaled_population[i] - mean_vector[i]) ** 2))

    diversity = np.mean(distances)
    #print(f"diversity={diversity}")
    return diversity

def computeHMCRandPAR(HM, ite):
    population = np.array([h for h, _ in HM])
    #print(f"population=\n{population}")
    div = compute_diversity(population)
    # print(f"==========================================================diversity={div:.4f}")
    #print(f"harmony memory: {HM}")
    
    scaled_div = scale_to_fuzzy_range(div, 5*np.sqrt(len(HM)))
    scaled_ite = scale_to_fuzzy_range(ite, NI)

    hmcr_par_sim.input['Iterations'] = scaled_ite
    hmcr_par_sim.input['Diversity'] = scaled_div
    #print(f"=========================================================={RED}ite={scaled_ite:.4f}, div={scaled_div:.4f}{RESET}")
    hmcr_par_sim.compute()

    hmcr = hmcr_par_sim.output['HMCR'] / 10  # scale 0-1
    par = hmcr_par_sim.output['PAR'] / 10
    #print(f"=========================================================={YELLOW}HMCR={hmcr:.2f}, PAR={par:.2f}{RESET}")
    return hmcr, par

def fuzzify_features(X_cpu):
    """Convert numerical features to fuzzy membership values"""
    X_cpu = X_cpu.to_numpy()  # move to CPU for fuzzy logic
    memberships = []

    for feat in X_cpu.T:
        feat = (feat - feat.mean()) / feat.std()  # z-score Standardize feature
        lo, hi = feat.min(), feat.max()
        med = lo + (hi - lo) / 2

        # Create triangular membership functions
        mf_lo = fuzz.trimf(feat, [lo, lo, med])
        mf_med = fuzz.trimf(feat, [lo, med, hi])
        mf_hi = fuzz.trimf(feat, [med, hi, hi])

        memberships.extend([mf_lo, mf_med, mf_hi])

    return cp.asarray(np.vstack(memberships).T)  # back to GPU

   

si = 0
results = []
while( si < len(seeds)):
    RANDOM_SEED = seeds[si]
    print(f"= RANDOM SEED {RANDOM_SEED} =")
    si += 1
   
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    cp.random.seed(RANDOM_SEED)

    CYAN = "\033[36m"
    DIM = "\033[2m"
    #==============COLOURS========================================
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    # Use WSL path (note: forward slashes)
    file_path = f'/mnt/c/Users/ralin/Documents/RF_HS_FF/data/{dataset}.csv'
    df = cudf.read_csv(file_path)

    _class = "Class"
    if dataset == "creditcard":
        _class = "Class"
        df[_class] = df[_class].astype('int64')
    elif dataset == "anemia":
        _class = "Diagnosis"
        unique_diagnoses = df[_class].unique().to_numpy()
        diagnosis_map = {diag: i for i, diag in enumerate(unique_diagnoses)}
        df[_class] = df[_class].map(diagnosis_map)
    elif dataset == "heart":
        _class = "HeartDisease"
        df = cudf.get_dummies(df)
        df[_class] = df[_class].astype('int64') 
        
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=RANDOM_SEED)

    #==============================noise
    y = df[_class]
    X = df.drop(_class, axis=1)
    noise = 0.2
    for col in X.columns:
        std = X[col].std()
        noise_col = cp.random.normal(0, noise*std, len(X))
        noise_col_df = cudf.DataFrame(noise_col, columns=[col])
        X[col] += noise_col

    version = 0
    while version < 3:
        version += 1
        print(f"{BLUE}== VERSION {version} =rs={RANDOM_SEED}={RESET}")

        if version == 3:
            print("Applying fuzzification...")
            X = fuzzify_features(X)

        maxf1mean = 0
        it_update = 0
        def rf_fitness(harmony, ite):
            global it_update
            global maxf1mean
            n_estimators = int(harmony[0])
            max_depth = int(harmony[1])
            max_features = float(harmony[2])

            clf = cuRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                
                random_state=RANDOM_SEED
            )

            if version == 1 or version == 2:
                X_np, y_np = X.to_numpy(), y.to_numpy()
            else:
                X_np, y_np = X.get(), y.to_numpy()
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            f1s = []

            for train_idx, val_idx in skf.split(X_np, y_np):
                X_train, X_val = X_np[train_idx], X_np[val_idx]
                y_train, y_val = y_np[train_idx], y_np[val_idx]

                clf.fit(X_train, y_train)
                preds = clf.predict(X_val)

                if dataset == "creditcard":
                    f1 = f1_score(y_val, preds)
                else:
                    f1 = f1_score(y_val, preds, average='macro')
               
                f1s.append(float(f1))

            currentf1mean = np.mean(f1s)
            if currentf1mean > maxf1mean:
                maxf1mean = currentf1mean
                it_update = ite
                print(f"{GREEN}{ite}: Estimators={n_estimators}, Depth={max_depth}, Features={max_features:.2f} → F1={currentf1mean:.4f} <-----------------------------------------------------------------------------{RESET}")
            else:
                print(f"{YELLOW}{ite}{RESET}, {CYAN}{maxf1mean:.4f}{RESET}: {DIM}Estimators={n_estimators}, Depth={max_depth}, Features={max_features:.2f} →{RESET} F1={currentf1mean:.4f}")

            return 1 - np.mean(f1s)

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
            
                if version == 1 or version == 3:
                    hmcr = 0.6  
                    par = 0.8  
                else:
                    hmcr, par = computeHMCRandPAR(HM, ite)
                #print(f"=========================================================={YELLOW}HMCR={hmcr:.2f}, PAR={par:.2f}{RESET}")
            
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

            return best[0], best[1]
        

        var_bounds = [
            (10, 100, 'int'),
            (5, 27, 'int'),
            (0.1, 1, 'float') #adjusted from 0.35
        ]

        best_harmony, best_fitness = harmony_search(
            obj_func=rf_fitness,
            var_bounds=var_bounds,
            hms=5,
            iterations=NI
        )

        n, d, f = best_harmony
        final_clf = cuRF(n_estimators=int(n), max_depth=int(d), max_features=float(f),random_state=RANDOM_SEED)

        if version == 1 or version == 2:
            X_np, y_np = X.to_numpy(), y.to_numpy()
        else:
            X_np, y_np = X.get(), y.to_numpy()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

        f1s, recalls, macro_f1s, weighted_recalls, weighted_f1s = [], [], [], [], []
        for train_idx, val_idx in skf.split(X_np, y_np):
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]

            final_clf.fit(X_train, y_train)
            preds = final_clf.predict(X_val)

            weighted_f1 = f1_score(y_val, preds, average='weighted')
            macro_f1 = f1_score(y_val, preds, average='macro')
            weighted_recall = recall_score(y_val, preds, average='weighted')

            weighted_f1s.append(weighted_f1)
            macro_f1s.append(macro_f1)
            weighted_recalls.append(weighted_recall)

        
            f1_per_class = f1_score(y_val, preds, average=None)
            recall_per_class = recall_score(y_val, preds, average=None)
            print(f"Per-class F1: {f1_per_class}")
            print(f"Per-class Recall: {recall_per_class}")

            f1s.append(float(f1_per_class[1]))
            recalls.append(float(recall_per_class[1]))
        if ( dataset == "anemia"):       
            print(f"FINAL → Weighted F1={np.mean(weighted_f1s):.4f}, Macro F1={np.mean(macro_f1s):.4f}, Weighted Recall={np.mean(weighted_recalls):.4f}")
            results.append({"version": version, "randomseed": RANDOM_SEED, "n": int(n), "d": int(d), "max_feat": f, "fitness":1-best_fitness, "f1": np.mean(macro_f1s), "recall":np.mean(weighted_recalls), "it_update":it_update})
        else:
            print(f"FINAL → F1={np.mean(f1s):.4f}, Macro F1={np.mean(macro_f1s):.4f},  Recall={np.mean(recalls):.4f}")
            results.append({"version": version, "randomseed": RANDOM_SEED,  "n": int(n), "d": int(d), "max_feat": f, "fitness":1-best_fitness, "f1": np.mean(f1s), "recall":np.mean(recalls), "it_update":it_update})

import json
print(json.dumps(results, default=float, indent=2))
 