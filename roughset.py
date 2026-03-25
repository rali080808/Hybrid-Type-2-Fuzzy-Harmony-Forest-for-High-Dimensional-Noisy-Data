from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

# dataset = "anemia"
# file_path = f'C:\\Users\\ralin\\Documents\\RF_HS_FF\\data\\{dataset}.csv'
# df = pd.read_csv(file_path)

# _class = "Diagnosis"
# unique_diagnoses = df[_class].unique()#.to_numpy()
# diagnosis_map = {diag: i for i, diag in enumerate(unique_diagnoses)}
# df[_class] = df[_class].map(diagnosis_map)
# # Get features
# features = df.drop(_class, axis=1).columns.tolist()
# print(f"Total features: {len(features)}")
# decision = _class

def discretize_features(df, features, decision, n_bins):
    print("Discretizing features...")
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df_discretized = df.copy()
    df_discretized[features] = discretizer.fit_transform(df[features])
    
    for col in features:
        df_discretized[col] = df_discretized[col].astype(int)
    print(df[features[:5] + [decision]].head(10))
    print(df_discretized[features[:5] + [decision]].head(10))
    return df_discretized

def positive_region(df, feature_subset, decision):
    if not feature_subset:
        return 0
    
    grouped = df.groupby(feature_subset)
    pos = 0
    
    for _, group in grouped:
        if len(group[decision].unique()) == 1:
            pos += len(group)
    
    return pos

def dependency(df, feature_subset, decision):
    pos = positive_region(df, feature_subset, decision)
    return pos / len(df)

def greedy_reduct(df, features_list, decision, max_features=15):
    
    # Discretize continuous features into bins
    df_discretized = discretize_features(df, features_list, decision, n_bins=5)

    full_dep = dependency(df_discretized, features_list, decision)
    
    target_dep = min(full_dep-0.05, 0.95)
    
    print(f"Full dependency: {full_dep:.4f}")
    print(f"Target dependency: {target_dep:.4f}")
    
    iteration = 0
    current_dep = 0.0
    selected = []
    while (current_dep < target_dep and len(selected) < len(features_list)) or len(selected) < 7:
        iteration += 1
        best_feature = None
        best_gain = -1
        
        print(f"\nIteration {iteration}: Current dep = {current_dep:.4f}, Selected = {len(selected)}")
        
        for f in features_list:
            if f not in selected:
                new_subset = selected + [f]
                gain = dependency(df_discretized, new_subset, decision)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
        
        if best_feature:
            selected.append(best_feature)
            current_dep = best_gain
            print(f"✓ Added '{best_feature}', new dep = {best_gain:.4f}")
            
            if len(selected) >= max_features:
                break
        else:
            break
    
    return selected
# greedy_reduct(df, features, decision, max_features=20)
# if len(df) > 50000:
#     df_sample = df_discretized.sample(n=50000, random_state=42)
#     print(f"Using sample of 50,000 rows for faster computation")
#     reduct = greedy_reduct(df_sample, features, decision, max_features=20)
# else:
#     reduct = greedy_reduct(df_discretized, features, decision, max_features=20)

# print("\n" + "="*60)
# print("RESULTS")
# print("="*60)
# print(f"Selected features ({len(reduct)}):")
# for i, feat in enumerate(reduct, 1):
#     print(f"  {i}. {feat}")
# print(f"Selected features ({reduct}):")
# final_dep = dependency(df_discretized, reduct, decision)
# full_dep = dependency(df_discretized, features, decision)
# print(f"\nDependency on discretized data: {final_dep:.4f}")
# print(f"Full dependency: {full_dep:.4f}")