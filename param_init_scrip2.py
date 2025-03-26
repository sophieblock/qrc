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

def update_pickle_file(file_path, test_key, t_steps, num_sample_states_L):
    gate_name = str(file_path).split('/')[-2]
    # Determine the target output file.
    rel_path = file_path.relative_to(base_dir)
    if updated_base_dir == base_dir:
        output_file = file_path
    else:
        output_file = updated_base_dir / rel_path

    # Check if the target file already exists and is updated.
    if output_file.exists():
        with open(output_file, 'rb') as f:
            updated_data = pickle.load(f)
        if isinstance(updated_data, pd.DataFrame):
            if len(updated_data) != 1:
                raise ValueError("Expected a single-row DataFrame, but found multiple rows.")
            updated_row = updated_data.iloc[0].to_dict()
        elif isinstance(updated_data, dict):
            updated_row = updated_data
        else:
            raise TypeError("Updated file data is neither a DataFrame nor a dictionary.")

        # Use the target file's keys for the update check.
        if updated_row.get('test_key') is not None and updated_row.get('DQFIM_stats_local') is not None:
            print(f"File {output_file} already updated. Skipping update.")
            return
    print(f"Processing {test_key}, gate {gate_name}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # If data is a DataFrame, assume it has a single row; convert it to a dictionary.
    if isinstance(data, pd.DataFrame):
        if len(data) != 1:
            raise ValueError("Expected a single-row DataFrame, but found multiple rows.")
        row = data.iloc[0].to_dict()
    elif isinstance(data, dict):
        row = data
    else:
        raise TypeError("Loaded pickle is neither a DataFrame nor a dictionary.")

    # Check if this file has already been updated by verifying the keys.
    # if row.get('test_key') is not None and row.get('DQFIM_stats_local') is not None:
    #     print(f"File {file_path} already updated. Skipping update.")
    #     return

    # Perform the computation and get updated values.
    fixed_params = row['fixed_params']
    N_ctrl = row['controls']
    N_reserv = row['reservoirs']
    qrc = Sim_QuantumReservoir(fixed_params, N_ctrl, N_reserv, N_ctrl * N_reserv, t_steps)
    gate = pickle.loads(base64.b64decode(row['Gate']))

    X = row['training_states']
    params = row['init_params']
    opt_params = row['opt_params']
    init_grads = row['init_grads']
    dqfim_original_dict = row.get('DQFIM_stats', {})
    print(f"dqfim_original_dict: {dqfim_original_dict.get('raw_trace')}")
    
    L_from_file = dqfim_original_dict.get('L', None)
    num_L = len(L_from_file)
    assert num_L == 20, f"Expected NUM_L==20, got {num_L}"

    dqfim_initial_lr_train, dqfim_dict_train, dqfim_dict_target = get_initial_learning_rate_DQFIM(
        params=params,
        qrc=qrc,
        X=X,
        gate=gate,
        init_grads=init_grads,
        compute_both=True
    )
    new_trace_target = dqfim_dict_target.get("trace_target", 0)
    evals_target = dqfim_dict_target.get("eigvals_target")
    nonzero_evals_target = evals_target[evals_target > threshold]
    new_var_target = float(np.var(nonzero_evals_target)) if len(nonzero_evals_target) > 0 else 0.0
    print(f"Computed Target DQFIM: Trace = {new_trace_target:.3f}, Var = {new_var_target:.3f}")

    target_stats = row["target DQFIM stats"]
    eigvals = target_stats.get('dqfim_eigvals', None)
    t_trace = sum(eigvals)
    nonzero = eigvals[eigvals > threshold]
    t_var = np.var(nonzero)
    print(f"Target DQFIM Stats read from file: Trace = {t_trace:.3f}, Var = {t_var:.3f}")

    computed_trace = float(jnp.real(new_trace_target))
    assert np.isclose(computed_trace, t_trace, atol=1e-3), \
        f"Computed target trace ({computed_trace:.3f}) not close to stored ({t_trace:.3f})"
    assert np.isclose(new_var_target, t_var, atol=1e-3), \
        f"Computed target variance ({new_var_target:.3f}) not close to stored ({t_var:.3f})"
    assert not np.isclose(computed_trace, 0.0)

    # Build a dictionary of updates.
    updates = {
        f'DQFIM_stats_{num_sample_states_L}_L_states': dqfim_original_dict,
        'test_key': test_key,
        'time_steps': t_steps,
        'trotter_step': t_steps,
        'DQFIM_stats_local': dqfim_dict_train,
    }
    # Update the row dictionary.
    row.update(updates)

    # Verify that the keys have been set.
    for key in updates:
        if row.get(key) is None:
            raise KeyError(f"Key '{key}' was not updated in file {file_path}")
        # else:
        #     print(f"SUCCESS: Key '{key}' updated.")

    # Now, if the original object was a DataFrame, recreate a single-row DataFrame;
    # otherwise, use the updated dictionary directly.
    # i.e. Recreate the updated object in its original format.
    if isinstance(data, pd.DataFrame):
        updated_obj = pd.DataFrame([row])
    else:
        updated_obj = row

    # # Determine the output file path.
    # # Use updated_base_dir if it is different from base_dir.
    # rel_path = file_path.relative_to(base_dir)
    # if updated_base_dir == base_dir:
    #     output_file = file_path
    # else:
    #     output_file = updated_base_dir / rel_path
    #     output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(updated_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Updated file saved: {output_file}\n")

    # Final verification: reload and check the updated keys.
    with open(output_file, 'rb') as f:
        updated_data = pickle.load(f)
    if isinstance(updated_data, pd.DataFrame):
        updated_row = updated_data.iloc[0].to_dict()
    else:
        updated_row = updated_data
    for key in updates:
        if updated_row.get(key) is None:
            raise KeyError(f"Updated file {output_file} row is missing key '{key}'!")
if __name__ == "__main__":
    N_ctrl = 1
    N_r = 1
    time_steps = 2
    N_train = 10
    num_epochs = 500
    num_sample_states_L = 20
    sample_range_label = "normal_.5pi"
    fixed_param_name = "fixed_params0"
    NUM_L = 20  # used for dqfim stats from file
    # Specify which test key(s) to process:
    trainable_test_keys = ['test0','test3', 'test6', 'test8', 'test9']
    trainable_test_keys = ['test0']

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
                update_pickle_file(file_path, test_key, t_steps=time_steps,num_sample_states_L=num_sample_states_L)
