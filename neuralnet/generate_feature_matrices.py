import numpy as np
from multiprocessing import Pool, cpu_count
import os

from constants import *
from burgers_solution import BurgersSolution


def get_training_data_folder_names():
    return sorted([item.name for item in DEFAULT_TRAINING_DATA_DIR.iterdir() if item.is_dir() and (item / METADATA_FILENAME).exists()])


def requires_artificial_viscosity(dx, u_i_minus_1, u_i_plus_1):
    ux = (u_i_plus_1 - u_i_minus_1) / (2.0 * dx)
    return ux < 0


def reverse_engineer_cq(dt, dx, u_i, u_next_i, u_i_minus_1, u_i_plus_1, nu=0):    
    # Compute ux for all points (vectorized)
    ux = (u_i_plus_1 - u_i_minus_1) / (2.0 * dx)
    
    # Only compute cq where ux < 0 (requires artificial viscosity)
    requires_av = ux < 0
    
    # Initialize cq array with None values (as nan for now)
    if isinstance(u_i, np.ndarray):
        cq_result = np.full_like(u_i, np.nan, dtype=float)
    else:
        # Scalar case
        if not requires_av:
            return None
        cq_result = None

    if isinstance(u_i, np.ndarray):
        # Vectorized path: u_i, u_next_i, etc. are arrays
        numerator = (
            u_next_i - u_i
            + u_i * (dt / (2 * dx)) * (u_i_plus_1 - u_i_minus_1)
            - nu * (dt / dx**2) * (u_i_plus_1 - 2 * u_i + u_i_minus_1)
        )

        denominator = (
            np.abs(ux) 
            * dt
            * (u_i_plus_1 - 2 * u_i + u_i_minus_1)
        )

        # Only compute where ux < 0
        valid_mask = (np.abs(denominator) >= CQ_DENOMINATOR_EPSILON) & requires_av
        
        cq_array = np.full_like(u_i, np.nan, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            cq_array[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        # Apply magnitude threshold
        cq_result = cq_array.copy()
        cq_result[(cq_array < 0) | (cq_array >= CQ_MAX_MAGNITUDE)] = np.nan
        
        return cq_result
    else:
        # Scalar path (original logic)
        numerator = (
            u_next_i - u_i
            + u_i * (dt / (2 * dx)) * (u_i_plus_1 - u_i_minus_1)
            - nu * (dt / dx**2) * (u_i_plus_1 - 2 * u_i + u_i_minus_1)
        )

        denominator = (
            abs(ux)
            * dt
            * (u_i_plus_1 - 2 * u_i + u_i_minus_1)
        )

        if abs(denominator) < CQ_DENOMINATOR_EPSILON:
            return None

        cq = numerator / denominator

        if not np.isfinite(cq):
            return None

        if cq < 0 or cq >= CQ_MAX_MAGNITUDE:
            return None

        return cq


def get_feature_matrices_for_sample(sample_name, seed):
    import time
    start_sample = time.time()
    
    X_rows = []
    y_rows = []

    burgers_solution = BurgersSolution(sample_name)
    coarseness_multiplier = int(np.random.default_rng(seed).choice(COARSENESS_MULTIPLIERS))
    nu = burgers_solution.nu
    fine_dt = burgers_solution.time_step_size
    fine_dx = burgers_solution.spatial_step_size
    coarse_dx = fine_dx * coarseness_multiplier
    coarse_dt_advection = (ALPHA * coarse_dx) / burgers_solution.max_u
    coarse_dt_diffusion = (BETA * (coarse_dx**2)) / burgers_solution.nu
    coarse_dt = min(coarse_dt_advection, coarse_dt_diffusion)

    print(f"Fine: dt={fine_dt:.6e}, dx={fine_dx:.6e} | Coarse: dt={coarse_dt:.6e}, dx={coarse_dx:.6e} | coarseness={coarseness_multiplier}")

    coarse_num_timesteps = int(burgers_solution.max_time // coarse_dt)
    coarse_num_domain_points = int(burgers_solution.domain_length // coarse_dx)

    # Randomly sample timesteps, then from each timestep randomly sample spatial points
    rng = np.random.default_rng(seed)
    all_timesteps = np.arange(coarse_num_timesteps - 1)
    sampled_timesteps = rng.choice(all_timesteps, size=min(MAX_TIMESTEPS_PER_SOLUTION, len(all_timesteps)), replace=False)
    all_spatial_steps = np.arange(U_RADIUS, coarse_num_domain_points - U_RADIUS)

    # Vectorized loop: process all spatial samples for a timestep in one shot
    for time_step in sampled_timesteps:
        sampled_spatial_steps = rng.choice(all_spatial_steps, size=min(MAX_SPATIAL_POINTS_PER_TIMESTEP, len(all_spatial_steps)), replace=False)
        
        # Current and next time indices (in actual time, not grid indices)
        curr_time = time_step * coarse_dt
        next_time = (time_step + 1) * coarse_dt
        
        # Convert actual time to fine-grid indices (same logic as BurgersSolution.get_u)
        t_index_float = curr_time / fine_dt
        t_index_lower = int(np.floor(t_index_float))
        t_index_upper = int(np.ceil(t_index_float))
        if t_index_upper >= burgers_solution.time_steps:
            t_index_upper = burgers_solution.time_steps - 1
        
        t_next_index_float = next_time / fine_dt
        t_next_index_lower = int(np.floor(t_next_index_float))
        t_next_index_upper = int(np.ceil(t_next_index_float))
        if t_next_index_upper >= burgers_solution.time_steps:
            t_next_index_upper = burgers_solution.time_steps - 1
        
        weight = t_index_float - t_index_lower
        weight_next = t_next_index_float - t_next_index_lower
        
        # Optimization 1: Direct integer indexing (no np.interp)
        # Since coarse_dx = fine_dx * COARSENESS_MULTIPLIER, spatial_step indices directly map to fine grid
        fine_indices = (sampled_spatial_steps[:, np.newaxis] + np.arange(-U_RADIUS, U_RADIUS + 1)) * coarseness_multiplier
        fine_indices = fine_indices.astype(int)
        
        # Fetch u values from fine grid (no interpolation needed since indices are integer-aligned)
        u_curr_lower = burgers_solution._u[t_index_lower, fine_indices]
        if t_index_lower == t_index_upper:
            u_curr_all = u_curr_lower
        else:
            u_curr_upper = burgers_solution._u[t_index_upper, fine_indices]
            u_curr_all = u_curr_lower * (1 - weight) + u_curr_upper * weight
        
        # Same for next time
        u_next_lower = burgers_solution._u[t_next_index_lower, fine_indices]
        if t_next_index_lower == t_next_index_upper:
            u_next_all = u_next_lower
        else:
            u_next_upper = burgers_solution._u[t_next_index_upper, fine_indices]
            u_next_all = u_next_lower * (1 - weight_next) + u_next_upper * weight_next
        
        # Optimization 2: Fully vectorized inner loop
        # u_curr_all shape: (num_spatial_points, 2*U_RADIUS+1)
        # Extract center, minus, plus directly as arrays
        u_i = u_curr_all[:, U_RADIUS]
        u_i_minus_1 = u_curr_all[:, U_RADIUS - 1]
        u_i_plus_1 = u_curr_all[:, U_RADIUS + 1]
        u_next_i = u_next_all[:, U_RADIUS]
        
        # Compute ux vectorized
        ux_array = (u_i_plus_1 - u_i_minus_1) / (2.0 * coarse_dx)
        
        # Call reverse_engineer_cq on full array
        cq_array = reverse_engineer_cq(coarse_dt, coarse_dx, u_i, u_next_i, u_i_minus_1, u_i_plus_1, nu)
        
        # Filter valid entries: requires_av (ux < 0) AND cq valid
        valid_mask = (ux_array < 0) & ~np.isnan(cq_array) & ~np.isinf(cq_array)
        valid_indices = np.where(valid_mask)[0]
        
        # Build batch for valid entries only (vectorized)
        if len(valid_indices) > 0:
            del_u_del_x = ux_array[valid_indices]
            del_u_del_x_squared = (u_i_plus_1[valid_indices] - 2 * u_i[valid_indices] + u_i_minus_1[valid_indices]) / (coarse_dx**2)
            cq_valid = cq_array[valid_indices]
            
            # Build feature array
            # order is dt, dx, dx, dxx, u's, nu
            n_valid = len(valid_indices)
            batch_X = np.column_stack([
                np.full(n_valid, coarse_dt),
                np.full(n_valid, coarse_dx),
                u_curr_all[valid_indices, U_RADIUS][:, None], # velocity at current point only
                u_curr_all[valid_indices, :U_RADIUS] - u_curr_all[valid_indices, :][:, [U_RADIUS]], # normalized stencil (subtract center point, middle term removed because always 0)
                u_curr_all[valid_indices, (U_RADIUS + 1):] - u_curr_all[valid_indices, :][:, [U_RADIUS]],
                np.full(n_valid, nu)
            ])
            batch_y = cq_valid
            
            X_rows.extend(batch_X)
            y_rows.extend(batch_y)

    elapsed = time.time() - start_sample
    print(f"  {sample_name}: {len(y_rows)} valid cq points extracted in {elapsed:.2f}s")
    
    return np.array(X_rows), np.array(y_rows)


if __name__ == "__main__":
    import time
    start_total = time.time()
    
    samples = get_training_data_folder_names()

    print(f"Found {len(samples)} samples. Extracting features...")
    seeds = [SEED + i for i in range(len(samples))]
    
    # Use chunksize to reduce worker overhead (chunks tasks to workers instead of one task per worker)
    chunksize = max(1, len(samples) // (cpu_count() * 4))  # Aim for ~4x more chunks than workers
    print(f"Using chunksize={chunksize} for {cpu_count()} CPUs")
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(get_feature_matrices_for_sample, zip(samples, seeds), chunksize=chunksize)

    X_matrices, y_matrices = zip(*results)

    X = np.vstack([m for m in X_matrices if m.size > 0])
    y = np.concatenate([m for m in y_matrices if m.size > 0])

    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"y mean: {y.mean():.4f}, std: {y.std():.4f}, min: {y.min():.4f}, max: {y.max():.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(FEATURE_MATRICES_PATH, X=X, y=y)
    print(f"Saved to {FEATURE_MATRICES_PATH}")
    
    elapsed_total = time.time() - start_total
    print(f"=== Total time: {elapsed_total:.2f}s ===" )
