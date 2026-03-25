import skfuzzy as fuzz
import numpy as np
import cudf

def fuzzy2_preprocess(X_gpu, y_gpu, n_terms=2, verbose=False):
 

    X_np = X_gpu.to_numpy()
    col_names = list(X_gpu.columns)
    new_cols = {}

    for col_idx, col_name in enumerate(col_names):
        feature = X_np[:, col_idx].astype(np.float64)
        f_min, f_max = feature.min(), feature.max()

        if f_max == f_min:
            # Degenerate feature — skip fuzzy expansion
            if verbose:
                print(f"  Skipping degenerate feature: {col_name}")
            continue

                # Evenly space n_terms centers across [min, max]
                #centers = np.linspace(f_min, f_max, n_terms)
        f_mean = feature.mean()
        f_std  = feature.std()

        centers = np.linspace(f_mean - f_std, f_mean + f_std, n_terms)
        sigma   = (2 * f_std) / (n_terms * 1.5)
        #sigma = (f_max - f_min) / (n_terms * 1.5)  # base width

        # Type-2 uncertainty: ±10% band on sigma
        sigma_lower = sigma * 0.90
        sigma_upper = sigma * 1.10

        for t_idx, center in enumerate(centers):
            label = f"{col_name}_t2_term{t_idx}"

            # Lower and upper Gaussian MFs
            mu_lower = np.exp(-0.5 * ((feature - center) / sigma_upper) ** 2)
            mu_upper = np.exp(-0.5 * ((feature - center) / sigma_lower) ** 2)

            # Crisp representative: midpoint of the interval
            mu_crisp = (mu_lower + mu_upper) / 2.0

            new_cols[label] = mu_crisp.astype(np.float32)

            if verbose:
                print(f"  {label}: center={center:.3f}, σ∈[{sigma_lower:.3f},{sigma_upper:.3f}]")

    # Build new cuDF DataFrame from new fuzzy columns and concat
    if new_cols:
        fuzzy_df = cudf.DataFrame(new_cols)
        X_fuzzified = cudf.concat([X_gpu.reset_index(drop=True),
                                   fuzzy_df.reset_index(drop=True)], axis=1)
    else:
        X_fuzzified = X_gpu

    if verbose:
        print(f"  Fuzzy2 expanded: {X_gpu.shape[1]} → {X_fuzzified.shape[1]} features")

    return X_fuzzified