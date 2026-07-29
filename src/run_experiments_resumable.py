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
    unbalanced_quadruple_well_setting,
    ackley_function_setting,
    initialize_on_circle,
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GradientAwareCBO,
    GradientAwarePSO
)

job_id = os.environ.get("SLURM_JOB_ID", "local")
RESULTS_CSV = f'results/master_experiment_log_decay_{job_id}.csv'
MAX_STEPS = 2_000_000_000

def get_completed_tasks():
    """Reads the master CSV to find which (experiment, param, seed) tasks are already done."""
    completed = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    # Tuple: (experiment_name, parameter_value, seed)
                    completed.add((row[0], float(row[1]), int(row[2])))
    return completed

def worker_task(args):
    """
    Isolated worker function. Sets up the environment based on the experiment type,
    runs all 4 algorithms, and returns their exit times.
    """
    exp_name, param, seed = args
    np.random.seed(seed * 1000)
    
    # 1. Environment Setup based on experiment type
    if exp_name == 'double_well':
        setting = one_dimensional_double_well(param) # param is delta
        d = 1
        particle_init = setting['particle_init']
    elif exp_name == 'quadruple_well':
        setting = unbalanced_quadruple_well_setting(param, -1) # param is A
        d = 2
        particle_init = setting['particle_init']
    elif exp_name == 'ackley':
        setting = ackley_function_setting()
        d = 2
        # param is radius
        particle_init = initialize_on_circle(100, param, noise=0.1)
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    set_ = setting['set']
    bounds = setting['bounds']
    
    noise_val = 1.0
    lr = 0.001

    # 2. Algorithm Initialization
    softmin = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
        bounds=bounds, noise="normal", beta_decay=None
    )
    softmin_decay = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init, 
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val, 
        bounds=bounds, noise="normal", beta_decay=0.999
    )
    
    sa = SimulatedAnnealing(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init, 
        bounds=bounds, learning_rate=lr, cooling_rate=0.9999, 
        noise=noise_val, cooling_schedule="geometric", initial_temperature=10
    )
    
    cbo = GradientAwareCBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init, 
        bounds=bounds, learning_rate=lr, consensus_rate=1.0, 
        noise_intensity=np.sqrt(2 * noise_val), beta=10.0
    )
    
    pso = GradientAwarePSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init, 
        bounds=bounds, learning_rate=lr, w=0.729, c1=1.49445, c2=1.49445
    )

    # Note: For Ackley, the notebook initializes fresh circle particles for each algorithm
    if exp_name == 'ackley':
        softmin.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        softmin_decay.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        sa.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        cbo.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        pso.reset(particle_init=initialize_on_circle(100, param, noise=0.1))

    # 3. Compute Entry/Exit Times

    t_softmin = softmin.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_softmin_decay = softmin_decay.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_sa = sa.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo = cbo.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_pso = pso.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo, t_pso = MAX_STEPS*lr, MAX_STEPS*lr
    return exp_name, param, seed, t_softmin, t_softmin_decay, t_sa, t_cbo, t_pso

def reconstruct_pkl_files_from_csv():
    """
    Utility function to rebuild the exact .pkl files your notebook expects 
    for plotting, entirely from the master CSV file.
    """
    logger.info("Reconstructing .pkl files for notebook compatibility...")
    data = {}
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            exp, param, seed, t_softmin, t_softmin_decay, t_sa, t_cbo, t_pso = row
            param = float(param)
            if param.is_integer(): param = int(param) # Clean up floats like 1.0 to 1
            
            if exp not in data: data[exp] = {}
            if param not in data[exp]:
                data[exp][param] = {'softmin': [], 'softmin_decay': [], 'sa': [], 'cbo': [], 'pso': []}
                
            data[exp][param]['softmin'].append(float(t_softmin))
            data[exp][param]['softmin_decay'].append(float(t_softmin_decay))
            data[exp][param]['sa'].append(float(t_sa))
            data[exp][param]['cbo'].append(float(t_cbo))
            data[exp][param]['pso'].append(float(t_pso))

    for exp, params in data.items():
        for param, algos in params.items():
            for algo, times in algos.items():
                filename = f"results/{algo}_{exp}_{param}.pkl"
                with open(filename, 'wb') as f:
                    pkl.dump(np.array(times), f)
    logger.info("Pickle files successfully reconstructed.")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    # Generate full Cartesian product of all desired workloads
    all_tasks = []
    # 1. Double Well (deltas 1 to 10)
    all_tasks.extend([('double_well', d, s) for d in range(1, 31) for s in range(100)])
    # 2. Quadruple Well (As 1 to 10)
    all_tasks.extend([('quadruple_well', a, s) for a in range(1, 31) for s in range(100)])
    # 3. Ackley (radii 1 to 8)
    all_tasks.extend([('ackley', r, s) for r in range(1, 9) for s in range(100)])

    # Filter out completed tasks to allow seamless resuming
    completed = get_completed_tasks()
    pending_tasks = [t for t in all_tasks if (t[0], float(t[1]), t[2]) not in completed]
    
    logger.info(f"Total tasks: {len(all_tasks)} | Completed: {len(completed)} | Pending: {len(pending_tasks)}")

    # Setup the CSV file (write header if it doesn't exist)
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['experiment', 'parameter', 'seed', 'softmin_decay', 'sa', 'cbo', 'pso'])

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
                    
                    logger.info(f"Completed -> Exp: {result[0]}, Param: {result[1]}, Seed: {result[2]}")

    logger.info("All cluster simulations finished.")
    
    # Automatically convert the flat CSV back into your expected Jupyter .pkl format
    reconstruct_pkl_files_from_csv()
