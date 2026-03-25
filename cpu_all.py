from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import f1_score, recall_score , log_loss
from sklearn.model_selection import StratifiedKFold
import random
import os
from datetime import datetime

dataset = "anemia" # "creditcard" or "anemia" or heart, python_exam
nameoffile = "cpu_all"
seeds = [42, 6, 89, 65, 22]# 103, 202, 149, 190, 177]
NI = 30
BW = 0.6
noise = 0
Iterations = ctrl.Antecedent(np.arange(0, 11, 0.1), 'Iterations')
Diversity = ctrl.Antecedent(np.arange(0, 11, 0.1), 'Diversity')
HMCR = ctrl.Consequent(np.arange(0, 11, 1), 'HMCR')
PAR = ctrl.Consequent(np.arange(0, 11, 1), 'PAR')
'''
Selected features (3):
  1. V14
  2. V12
  3. V4

Dependency on discretized data: 0.9268
'''
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
    print(f"diversity={diversity}")
    return diversity

def computeHMCRandPAR(HM, ite):
    population = np.array([h for h, _ in HM])
    print(f"population=\n{population}")
    div = compute_diversity(population)
    # print(f"==========================================================diversity={div:.4f}")
    #print(f"harmony memory: {HM}")
    
    scaled_div = scale_to_fuzzy_range(div, 5*np.sqrt(len(HM)))
    scaled_ite = scale_to_fuzzy_range(ite, NI)

    hmcr_par_sim.input['Iterations'] = scaled_ite
    hmcr_par_sim.input['Diversity'] = scaled_div
    print(f"=========================================================={RED}ite={scaled_ite:.4f}, div={scaled_div:.4f}{RESET}")
    hmcr_par_sim.compute()

    hmcr = hmcr_par_sim.output['HMCR'] / 10  # scale 0-1
    par = hmcr_par_sim.output['PAR'] / 10
    print(f"=========================================================={YELLOW}HMCR={hmcr:.2f}, PAR={par:.2f}{RESET}")
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

    return np.asarray(np.vstack(memberships).T)  # back to GPU

   
imp = []
forest = []
feature_names = []
si = 0
results = []
CYAN = "\033[36m"
DIM = "\033[2m"
#==============COLOURS========================================
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"
while( si < len(seeds)):
    RANDOM_SEED = seeds[si]
    print(f"= RANDOM SEED {RANDOM_SEED} =")
    si += 1
   
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
     
    # Use WSL path (note: forward slashes)
    file_path = f'C:\\Users\\ralin\\Documents\\RF_HS_FF\\data\\{dataset}.csv'
    df = pd.read_csv(file_path)

    # DATA PREPROCESSING
    _class = "Class"
    if dataset == "creditcard":
        _class = "Class"
        df[_class] = df[_class].astype('int64')
    elif dataset == "anemia":
        _class = "Diagnosis"
        unique_diagnoses = df[_class].unique()#.to_numpy()
        diagnosis_map = {diag: i for i, diag in enumerate(unique_diagnoses)}
        df[_class] = df[_class].map(diagnosis_map)
    elif dataset == "heart":
        _class = "HeartDisease"
        df = pd.get_dummies(df)
        df[_class] = df[_class].astype('int64') 
    elif dataset == "python_exam":
        _class = "passed_exam"
        df = df.drop('country', axis=1)
        df = df.drop('final_exam_score', axis=1)
        df = df.drop('student_id', axis=1)
        df['prior_programming_experience'] = pd.Categorical(
            df['prior_programming_experience'],
            categories=['None', 'Beginner', 'Intermediate', 'Advanced'],
            ordered=True
        ).codes
        
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=RANDOM_SEED)

    #==============================noise
    y = df[_class]
    X = df.drop(_class, axis=1)
    feature_names = X.columns.tolist() 
    
    
    feature_names =  ['WBC', 'HGB', 'MCH']
    X = df[feature_names] # for reduct

   
    for col in X.columns:
        std = X[col].std()
        noise_col = np.random.normal(0, noise*std, len(X))
        noise_col_df = pd.DataFrame(noise_col, columns=[col])
        X[col] += noise_col
    
    #normalize features
    #X = (X - X.min()) / (X.max() - X.min())

    version = 0
    lossesperversion = []
    f1perversion = []
    while version < 3:
        version += 1
        print(f"{BLUE}== VERSION {version} =rs={RANDOM_SEED}={RESET}")
        lossesPerVersionIteration = [ ]
        f1perversioniteration = [ ]
        if version == 3:
            print("Applying fuzzification...")
            X = fuzzify_features(X)

        maxf1mean = 0
        it_update = 0
        bestloss = float('inf')
        def rf_fitness(harmony, ite):
            global it_update
            global maxf1mean
            global bestloss
            n_estimators = int(harmony[0])
            max_depth = int(harmony[1])
            max_features = float(harmony[2])

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                n_jobs=-1,
                random_state=RANDOM_SEED
            )

            if version == 1 or version == 2:
                X_np, y_np = X.to_numpy(), y.to_numpy()
            else:
                X_np, y_np = X, y.to_numpy()
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            f1s = []
            lossesKfold = []
            for train_idx, val_idx in skf.split(X_np, y_np):
                X_train, X_val = X_np[train_idx], X_np[val_idx]
                y_train, y_val = y_np[train_idx], y_np[val_idx]

                clf.fit(X_train, y_train)
                preds = clf.predict(X_val)

                if dataset == "creditcard" or dataset == "python_exam":
                    f1 = f1_score(y_val, preds)
                else:
                    f1 = f1_score(y_val, preds, average='macro')
                y_proba = clf.predict_proba(X_val)
                lossesKfold.append(log_loss(y_val, y_proba))
                
                f1s.append(float(f1))
            print(f"Log Loss: {np.mean(lossesKfold):.4f}")
            
            
            currentf1mean = np.mean(f1s)
            if np.mean(lossesKfold) < bestloss:
                bestloss = np.mean(lossesKfold)
                print(f"{BLUE}{ite}: Estimators={n_estimators}, Depth={max_depth}, Features={max_features:.2f} → Log Loss={bestloss:.4f} <-----------------------------------------------------------------------------{RESET}")
            if currentf1mean > maxf1mean:
                maxf1mean = currentf1mean
                it_update = ite
                print(f"{GREEN}{ite}: Estimators={n_estimators}, Depth={max_depth}, Features={max_features:.2f} → F1={currentf1mean:.4f} <-----------------------------------------------------------------------------{RESET}")
            else:
                print(f"{YELLOW}{ite}{RESET}, {CYAN}{maxf1mean:.4f}{RESET}: {DIM}Estimators={n_estimators}, Depth={max_depth}, Features={max_features:.2f} →{RESET} F1={currentf1mean:.4f}")

            lossesPerVersionIteration.append(bestloss)
            f1perversioniteration.append(maxf1mean)

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
        final_clf = RandomForestClassifier(n_estimators=int(n), max_depth=int(d), max_features=float(f), n_jobs=-1, random_state=RANDOM_SEED)

        if version == 1 or version == 2:
            X_np, y_np = X.to_numpy(), y.to_numpy()
        else:
            X_np, y_np = X, y.to_numpy()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

        f1s0, f1s1, recalls, macro_f1s, weighted_recalls, weighted_f1s = [], [], [], [], [], []
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

            f1s0.append(float(f1_per_class[0]))
            f1s1.append(float(f1_per_class[1]))
            recalls.append(float(recall_per_class[1]))
   
        # for the diagrams
        lossesperversion.append(lossesPerVersionIteration)
        f1perversion.append(f1perversioniteration)

        forest.append(final_clf)
        imp.append(final_clf.feature_importances_)
        if ( dataset == "anemia"):       
            print(f"FINAL → Weighted F1={np.mean(weighted_f1s):.4f}, Macro F1={np.mean(macro_f1s):.4f}, Weighted Recall={np.mean(weighted_recalls):.4f}")
            results.append({"version": version, "randomseed": RANDOM_SEED, "n": int(n), "d": int(d), "max_feat": f, "fitness":1-best_fitness, "f1": np.mean(macro_f1s), "recall":np.mean(weighted_recalls), "it_update":it_update})
        else:
            print(f"FINAL → F1_1={np.mean(f1s1):.4f}, F1_0={np.mean(f1s0):.4f}, Macro F1={np.mean(macro_f1s):.4f},  Recall={np.mean(recalls):.4f}")
            results.append({"version": version, "randomseed": RANDOM_SEED,  "n": int(n), "d": int(d), "max_feat": f, "fitness":1-best_fitness, "f1_0": np.mean(f1s0), "f1_1": np.mean(f1s1), "recall":np.mean(recalls), "it_update":it_update})
    plt.figure(figsize=(6,4))
    plt.plot(range(1, NI + 1+len(HM)), lossesperversion[0], label="HS, RF", color='blue')
    plt.plot(range(1, NI + 1+len(HM)), lossesperversion[1], label="FL, HS, RF", color='orange')
    plt.plot(range(1, NI + 1+len(HM)), lossesperversion[2], label="FF, HS, RF", color='green')
    plt.legend()
    plt.title(f"Log Loss по итерации - версия {version}, seed {RANDOM_SEED}, шум {noise}*std")
    plt.xlabel("Итерация")
    plt.ylabel("Log Loss")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(range(1, NI + 1+len(HM)), f1perversion[0], label="HS, RF", color='blue')
    plt.plot(range(1, NI + 1+len(HM)), f1perversion[1], label="FL, HS, RF", color='orange')
    plt.plot(range(1, NI + 1+len(HM)), f1perversion[2], label="FF, HS, RF", color='green')
    plt.legend()
    plt.title(f"F1 по итерации - seed {RANDOM_SEED}, шум {noise}*std")
    plt.xlabel("Итерация")
    plt.ylabel("F1 Score")
    plt.show()
import json
print(json.dumps(results, default=float, indent=2))
 

for i in range(len(seeds)):
    for v in range(1, 4):
        idxxxx = i*3 + (v - 1)
        TOP_K = 10

        fig, ax = plt.subplots(figsize=(7, 5))
        std = np.std([tree.feature_importances_ for tree in forest[idxxxx].estimators_], axis=0)
        if v == 3:
            fuzzy_feature_names = []
            for col in feature_names:
                fuzzy_feature_names.extend([f"{col}_lo", f"{col}_med", f"{col}_hi"])
            imp[idxxxx] = pd.Series(imp[idxxxx], index=fuzzy_feature_names)
             
        else:
            imp[idxxxx] = pd.Series(imp[idxxxx], index=feature_names)
        imp[idxxxx] = imp[idxxxx].sort_values(ascending=False).head(TOP_K)
        imp[idxxxx].plot.bar( ax=ax)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # plt.savefig(f"C:\\Users\\ralin\\Documents\\RF_HS_FF\\just_saved_diagrams\\cpu__feature_importances_{seeds[i]}_{v}_{noise}_{timestamp}.png", dpi=150)

print("Plotting F1 scores...")
f1_v1 = []
f1_v2 = []
f1_v3 = []
for res in results:
    if dataset == "anemia":
        if res["version"] == 1:
            f1_v1.append(res["f1"])
        elif res["version"] == 2:
            f1_v2.append(res["f1"])
        elif res["version"] == 3:
            f1_v3.append(res["f1"])
    else:
        if res["version"] == 1:
            f1_v1.append(res["f1_0"])
        elif res["version"] == 2:
            f1_v2.append(res["f1_0"])
        elif res["version"] == 3:
            f1_v3.append(res["f1_0"])
print(f"f1_v1={f1_v1}")

offset = 0.2
plt.figure(figsize=(8,5)) 
seeds = np.array(seeds)
plt.scatter(seeds - offset, f1_v1, label="HS, RF", s=80)
plt.scatter(seeds, f1_v2, label="FL, HS, RF", s=80)
plt.scatter(seeds + offset, f1_v3, label="FF, HS, RF", s=80)

plt.title(f"F1 Score при различни семена с шум {noise}*std на данните за aнемия")
plt.xlabel("Seed")
plt.ylabel("F1 Score")

plt.xticks(seeds)  # show seeds as labels
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()     
      