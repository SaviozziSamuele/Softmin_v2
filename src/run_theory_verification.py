import os
import csv
import logging
import multiprocessing as mp
import numpy as np
import pickle as pkl
import itertools

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from swarm_particle_optimization import (
    one_dimensional_double_well,
    SoftminEnergyAdagrad
)

job_id = os.environ.get("SLURM_JOB_ID", "local")
RESULTS_CSV = f'results/master_theory_verification_init0_0p05_{job_id}.csv'
MAX_STEPS = 10_000_000

def get_completed_tasks():
    """Reads the master CSV to find which (experiment, param, seed) tasks are already done."""
    completed = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    completed.add((row[0], float(row[1]), float(row[2]), int(row[3])))
    return completed

def worker_task(args):
    """
    Isolated worker function. Sets up the environment based on the experiment type.
    """
    experiment, param, noise_val, seed = args
    np.random.seed(seed * 1000)


    setting = one_dimensional_double_well(param) # param is delta
    d = 1
    particle_init = [[setting['critical_points'][0], setting['critical_points'][0]]]

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    set_ = setting['set']
    bounds = setting['bounds']

    lr = 0.01

    # 2. Algorithm Initialization
    softmin_decay = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
        bounds=bounds, noise="normal", beta_decay=None, beta=2.0
    )

    # 3. Compute Entry/Exit Times
    t_softmin = softmin_decay.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    return experiment, param, noise_val, seed, t_softmin

def reconstruct_pkl_files_from_csv():
    """
    Utility function to rebuild the exact .pkl files your notebook expects
    for plotting, entirely from the master CSV file.
    """
    logger.info("Reconstructing .pkl files for notebook compatibility...")
    data = {}
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            # Unpack the CSV row
            exp, param, noise_val, seed, t_soft = row

            # Clean up param
            param = float(param)
            if param.is_integer(): param = int(param)

            # Clean up noise_val
            noise_val = float(noise_val)
            if noise_val.is_integer(): noise_val = int(noise_val)

            # Build the nested dictionary: data[exp][param][noise_val][algo]
            if exp not in data:
                data[exp] = {}
            if param not in data[exp]:
                data[exp][param] = {}
            if noise_val not in data[exp][param]:
                data[exp][param][noise_val] = {'softmin_static': []}

            # Append the time
            data[exp][param][noise_val]['softmin_static'].append(float(t_soft))

    # Loop through the nested dictionary to save the files
    for exp, params in data.items():
        for param, noises in params.items():

            # Clean param for the filename
            clean_param = str(param).replace('.', 'p')

            for noise_val, algos in noises.items():

                # Clean noise_val for the filename
                clean_noise = str(noise_val).replace('.', 'p')

                for algo, times in algos.items():
                    # Generate the clean filename
                    filename = f"results/{algo}_{exp}_init0_{clean_param}_{clean_noise}.pkl"

                    with open(filename, 'wb') as f:
                        pkl.dump(np.array(times), f)

    logger.info("Pickle files successfully reconstructed.")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Generate full Cartesian product of all desired workloads
    all_tasks = []
    # 1. Double Well (deltas 1 to 10)
    all_tasks.extend([('double_well', d, 0.05, s) for d in np.arange(0.25,4.25,0.125) for s in range(100)])

    # Filter out completed tasks to allow seamless resuming
    completed = get_completed_tasks()
    pending_tasks = [t for t in all_tasks if (t[0], float(t[1]), float(t[2]), int(t[3])) not in completed]

    logger.info(f"Total tasks: {len(all_tasks)} | Completed: {len(completed)} | Pending: {len(pending_tasks)}")

    # Setup the CSV file (write header if it doesn't exist)
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['experiment', 'parameter', 'noise', 'seed', 'softmin_static'])

        if pending_tasks:
            N_CPU = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
            logger.info(f"Launching pool with {N_CPU} workers...")

            # Use imap_unordered to process tasks efficiently without blocking
            with mp.Pool(N_CPU) as pool:
                for result in pool.imap_unordered(worker_task, pending_tasks):
                    # Write results instantly to disk as they complete
                    writer.writerow(result)
                    csvfile.flush() # Force write to physical disk immediately
                    os.fsync(csvfile.fileno()) # Extreme protection against cluster hard-crashes

                    logger.info(f"Completed -> Exp: {result[0]}, Param: {result[1]}, Noise: {result[2]}, Seed: {result[3]}")

    logger.info("All cluster simulations finished.")

    # Automatically convert the flat CSV back into your expected Jupyter .pkl format
    reconstruct_pkl_files_from_csv()
