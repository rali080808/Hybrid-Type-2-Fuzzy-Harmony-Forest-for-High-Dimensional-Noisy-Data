import numpy as np
import cudf
import cupy as cp

def fuzzy2_preprocess(X_gpu, n_terms=2, verbose=False, 
                      uncertainty_type='std',  # 'std' or 'mean'
                      uncertainty_range=0.5):  # +=20% uncertainty by default
    
    X_np = X_gpu.to_numpy()
    col_names = list(X_gpu.columns)
    new_cols = {}
    skipped_features = 0

    for col_idx, col_name in enumerate(col_names):
        feature = X_np[:, col_idx].astype(np.float64)
        
      
        f_min = np.percentile(feature, 2)
        f_max = np.percentile(feature, 98)
        
         
        if f_max <= f_min or feature.max() == feature.min():
            if verbose:
                print(f"  Skipping degenerate feature: {col_name}")
            skipped_features += 1
            continue
        
       
        if n_terms == 1:
            centers = np.array([np.median(feature)])
        else:
            centers = np.linspace(f_min, f_max, n_terms)
        
         
        if n_terms > 1:
            center_spacing = (f_max - f_min) / (n_terms - 1)
            sigma = center_spacing * 0.6  
        else:
            sigma = (f_max - f_min) / 3.0
        
        sigma = max(sigma, 1e-6)
        
        if verbose:
            print(f"\n{col_name}:")
            print(f"  Range (2-98%): [{f_min:.3f}, {f_max:.3f}]")
            print(f"  Centers: {centers}")
            print(f"  Base sigma: {sigma:.4f}")
            print(f"  Uncertainty type: {uncertainty_type}")
        
        for t_idx, center in enumerate(centers):
            label = f"{col_name}_t2_term{t_idx}"
            
            if uncertainty_type == 'std':
                sigma_lower = sigma * (1 - uncertainty_range)
                sigma_upper = sigma * (1 + uncertainty_range)
                
                mu_upper = np.exp(-0.5 * ((feature - center) / sigma_lower) ** 2)
                mu_lower = np.exp(-0.5 * ((feature - center) / sigma_upper) ** 2)
                
            else:  
                mean_lower = center - sigma * uncertainty_range
                mean_upper = center + sigma * uncertainty_range
                
                mu_lower = np.exp(-0.5 * ((feature - mean_upper) / sigma) ** 2)
                mu_upper = np.exp(-0.5 * ((feature - mean_lower) / sigma) ** 2)
                
                mask = (feature >= mean_lower) & (feature <= mean_upper)
                mu_upper[mask] = 1.0
            
            mu_crisp = (mu_lower + mu_upper) / 2.0
            
            mu_crisp = np.nan_to_num(mu_crisp, nan=0.0, posinf=1.0, neginf=0.0)
            
            if mu_crisp.std() < 0.01:
                if verbose:
                    print(f"  WARNING: {label} has near-zero variance (std={mu_crisp.std():.6f}), skipping...")
                continue
            
            new_cols[label] = mu_crisp.astype(np.float32)
            
            if verbose:
                if uncertainty_type == 'std':
                    print(f"  {label}: center={center:.3f}, σ∈[{sigma_lower:.3f},{sigma_upper:.3f}], "
                          f"μ range=[{mu_crisp.min():.3f}, {mu_crisp.max():.3f}], std={mu_crisp.std():.3f}")
                else:
                    print(f"  {label}: σ={sigma:.3f}, mean∈[{mean_lower:.3f},{mean_upper:.3f}], "
                          f"μ range=[{mu_crisp.min():.3f}, {mu_crisp.max():.3f}], std={mu_crisp.std():.3f}")

    if new_cols:
        fuzzy_df = cudf.DataFrame(new_cols)
        
        for col in fuzzy_df.columns:
            fuzzy_df[col] = fuzzy_df[col].astype('float32')
        
        X_fuzzified = fuzzy_df  
        
        if verbose:
            print(f"\n  Fuzzy2 preprocessing complete:")
            print(f"    Original features: {X_gpu.shape[1]}")
            print(f"    Skipped features: {skipped_features}")
            print(f"    Fuzzy features created: {len(new_cols)}")
            print(f"    Final features: {X_fuzzified.shape[1]}")
            print(f"    Uncertainty type: {uncertainty_type}")
    else:
        X_fuzzified = X_gpu
        if verbose:
            print(f"\n  WARNING: No fuzzy features created, returning original data")

    return X_fuzzified

 
def fuzzy2_preprocess_concat(X_gpu, n_terms=2, verbose=False, 
                              uncertainty_type='std', uncertainty_range=0.2): 
    fuzzy_df = fuzzy2_preprocess(X_gpu, n_terms, verbose, uncertainty_type, uncertainty_range)
    
    if isinstance(fuzzy_df, cudf.DataFrame) and fuzzy_df.shape[1] > 0:
        X_combined = cudf.concat([X_gpu.reset_index(drop=True),
                                   fuzzy_df.reset_index(drop=True)], axis=1)
        if verbose:
            print(f"\n  Concatenated: {X_gpu.shape[1]} original + {fuzzy_df.shape[1]} fuzzy = {X_combined.shape[1]} total")
        return X_combined
    else:
        return X_gpu