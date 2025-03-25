import os
import pickle
import base64
from pathlib import Path
import pandas as pd
import jax
from jax import numpy as jnp
import numpy as np
# Import your functions and classes (adjust these imports as necessary)
from param_init_analog_new import (
    get_initial_learning_rate_DQFIM,
    Sim_QuantumReservoir
)
threshold = 1e-10

def update_pickle_file(file_path,test_key, t_steps):
    gate_name = str(file_path).split('/')[-2]
    print(f"Processing {test_key}, gate {gate_name}")
    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    updated = False
    # Loop over each row in the DataFrame (each run)
    for idx, row in df.iterrows():
        # We assume that runs with use_L==False do not yet have the new key.
        if 'DQFIM_stats_local' not in row or 'test_key' not in row:
            
            # Reinstantiate the quantum reservoir using stored fixed parameters and settings.
            fixed_params = row['fixed_params']
            N_ctrl_row = row['controls']         # stored as "controls"
            N_reserv_row = row['reservoirs']       # stored as "reservoirs"
            # t_steps = row['time_steps']          # stored as "trotter_step"
            qrc = Sim_QuantumReservoir(fixed_params, N_ctrl_row, N_reserv_row, N_reserv_row * N_ctrl_row, t_steps)
            print(f"--- Updating row {idx}, N_C = {N_ctrl_row}, N_R = {N_reserv_row}, T = {t_steps} ---")
            # Retrieve the target gate from the stored base64 encoded pickle
            gate = pickle.loads(base64.b64decode(row['Gate']))

            # Get training states, optimized parameters, and initial gradients from the stored data
            X = row['training_states']
            params = row['init_params']
            opt_params = row['opt_params']
            init_grads = row['init_grads']
            dqfim_original_dict = row.get('DQFIM_stats', {})
            L_from_file = dqfim_original_dict.get('L', None)
          
            num_L = len(L_from_file)
            assert num_L == 20, f"Expected NUM_L==20, got {num_L}"

            row[f'DQFIM_stats_{num_L}_L_states'] = dqfim_original_dict
            row['test_key'] = test_key
            row['time_steps'] = t_steps
            row['trotter_step'] = t_steps
           
            # Recalculate the DQFIM for the training space (i.e. target_DQFIM=False)
            dqfim_initial_lr_train, dqfim_dict_train,dqfim_dict_target = get_initial_learning_rate_DQFIM(
                params=params,
                qrc=qrc,
                X=X,
                gate=gate,
                init_grads=init_grads,
                compute_both=True
            )
            # Print computed target DQFIM stats (to three decimal places)
            new_trace_target = dqfim_dict_target.get("trace_target", 0)
            evals_target = dqfim_dict_target.get("eigvals_target")
           
            nonzero_evals_target = evals_target[evals_target > threshold]
            new_var_target = float(np.var(nonzero_evals_target)) if len(nonzero_evals_target) > 0 else 0.0
            
            print(f"Computed Target DQFIM: Trace = {new_trace_target:.3f}, Var = {new_var_target:.3f}")
            
            # print(f"dqfim_dict_train keys: {dqfim_dict_train.keys()}")
            # print(f"dqfim_dict_train.head(): {dqfim_dict_train.items()}")
            target_stats = row["target DQFIM stats"]
            # print(f"target_stats keys: {target_stats.keys()}, {type(target_stats)}")
            # print(f"target_stats.items(): {target_stats.items()}")
            eigvals = target_stats.get('dqfim_eigvals', None)
            t_trace = sum(eigvals)
            nonzero = eigvals[eigvals > threshold]
            t_var = np.var(nonzero)
            print(f"Target DQFIM Stats read from file: Trace = {t_trace:.3f}, Var = {t_var:.3f}")
            # Add assertion checks to ensure the computed and stored target stats are close.
            computed_trace = float(jnp.real(new_trace_target))
            assert np.isclose(computed_trace, t_trace, atol=1e-3), \
                f"Computed target trace ({computed_trace:.3f}) not close to stored ({t_trace:.3f})"
            assert np.isclose(new_var_target, t_var, atol=1e-3), \
                f"Computed target variance ({new_var_target:.3f}) not close to stored ({t_var:.3f})"
            assert not np.isclose(computed_trace, 0.0)
            # # 1. Original QFIM stats (from "QFIM Results")
            orig = row[f'DQFIM_stats_{num_L}_L_states']
            # print(f"orig.keys: {orig.keys()}")
            orig_trace = float(orig.get("raw_trace", 0))
            orig_var = float(orig.get("raw_var_nonzero", 0))
            print(f"DQFIM_stats_{num_L}_L_states (originally DQFIM_stats): Trace = {orig_trace:.3f}, Var = {orig_var:.3f}")
          
           
            
            
            # 3. File-stored DQFIM stats (from "DQFIM_stats")
            if "DQFIM_stats" in row:
                file_stats = row["DQFIM_stats"]
                f_trace = float(file_stats.get("raw_trace", 0))
                f_var = float(file_stats.get("raw_var_nonzero", 0))
                print(f"Original DQFIM Stats: Trace = {f_trace:.3f}, Var = {f_var:.3f}")
            else:
                print("Original DQFIM Stats: Not available")
            
            new_trace_train = dqfim_dict_train.get("trace_train", 0)
            evals_train = dqfim_dict_train.get("eigvals_train")
            nonzero_evals = evals_train[evals_train > threshold]
            new_var_train = float(np.var(nonzero_evals)) if len(nonzero_evals) > 0 else 0.0
            
            print(f"Computed Training DQFIM: Trace = {new_trace_train:.3f}, Var = {new_var_train:.3f}")
            row['DQFIM_stats_local'] = dqfim_dict_train
            
            updated = True
    if updated:
        # Save the updated DataFrame to a new directory preserving relative structure.
        # The updated files will be stored under updated_base_dir.
        rel_path = file_path.relative_to(base_dir)
        updated_file_path = updated_base_dir / rel_path
        updated_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(updated_file_path, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Updated file saved: {updated_file_path}\n")
    else:
        print(f"No update needed for file: {file_path}")
    
if __name__ == "__main__":
    N_ctrl = 1
    N_r = 1
    time_steps = 2
    N_train = 10
    num_epochs = 500
    sample_range_label = "normal_.5pi"
    fixed_param_name = "fixed_params0"
    NUM_L = 20  # used for dqfim stats from file
    # Specify which test key(s) to process:
    trainable_test_keys = ['test0','test3', 'test6', 'test8', 'test9']

    # Parameterize the base directory for original results
    base_dir = Path(f"./param_initialization_final/analog_results/Nc_{N_ctrl}/epochs_{num_epochs}")
    # Updated files will be saved under this directory:
    updated_base_dir = Path(f"./param_initialization_final/analog_results_updated/Nc_{N_ctrl}/epochs_{num_epochs}")

    if not base_dir.exists():
        print(f"Base directory {base_dir} does not exist. Please check the path.")
    else:
        for file_path in base_dir.rglob("*.pickle"):
            # Check if any of the trainable test keys exactly appear in the path parts.
            matching_keys = [key for key in trainable_test_keys if key in file_path.parts]
            if matching_keys:
                test_key = matching_keys[0]  # Use the first matching key.
                print(f"Processing file: {file_path} (test key: {test_key})")
                update_pickle_file(file_path, test_key, t_steps=time_steps)