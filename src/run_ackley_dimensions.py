import os
import csv
import logging
import multiprocessing as mp
import numpy as np
import pickle as pkl

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from swarm_particle_optimization import (
    high_dim_ackley_function_setting,
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GradientAwareCBO,
    GradientAwarePSO
)

RESULTS_CSV = 'results/master_high_dim_ackley_log_exp_on_dims.csv'
MAX_STEPS = 10_000_000

def get_completed_tasks():
    """Reads the master CSV to find which (dimension, seed) tasks are already complete."""
    completed = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    # Tuple: (dimension, seed)
                    completed.add((int(row[0]), int(row[1])))
    return completed

def worker_task(args):
    """
    Isolated worker function for the high-dimensional Ackley evaluation.
    """
    dimension, seed = args
    
    # 1. Establish deterministic behavior for this worker
    np.random.seed(seed * 1000)
    
    # 2. Retrieve dimensional setting
    setting = high_dim_ackley_function_setting(d=dimension)
    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    set_ = setting['set']
    bounds = setting['bounds']

    noise_val = 1.0
    lr = 0.01
    
    # 4. Instantiate Softmin with Beta Decay
    softmin_decay = SoftminEnergyAdagrad(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SoftMin_decay_ackley_function_dim{dimension}", 
        initial_lr=0.01, adaptive_lr=False, noise_intensity=1.0, 
        noise="normal", beta_decay=0.999
    )

    # 5. Instantiate Simulated Annealing
    sa = SimulatedAnnealing(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SimulatedAnnealing_ackley_function_dim{dimension}", 
        learning_rate=0.01, cooling_rate=0.9999, noise=1.0, 
        cooling_schedule="geometric", initial_temperature=10.0
    )

    cbo = GradientAwareCBO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0, 
        noise_intensity=np.sqrt(2 * noise_val), beta=10.0
    )
    
    pso = GradientAwarePSO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, w=0.729, c1=1.49445, c2=1.49445
    )

    # 6. Execute vectorized step loops
    t_softmin_decay = softmin_decay.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_sa = sa.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo = cbo.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_pso = pso.first_entry_time(target_set=set_, max_steps=MAX_STEPS)

    return dimension, seed, t_softmin_decay, t_sa, t_cbo, t_pso

def reconstruct_pkl_files_from_csv():
    """
    Translates the fault-tolerant CSV log back into the segregated .pkl files 
    your Jupyter Notebook uses for rendering the log-mean plots.
    """
    logger.info("Reconstructing high-dimensional .pkl files for notebook compatibility...")
    data = {}
    
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            dimension, seed, t_soft_decay, t_sa, t_cbo, t_pso = row
            dimension = int(dimension)
            
            if dimension not in data:
                data[dimension] = {'softmin_decay': [], 'sa': [], 'cbo': [], 'pso': []}
                
            data[dimension]['softmin_decay'].append(float(t_soft_decay))
            data[dimension]['sa'].append(float(t_sa))
            data[dimension]['cbo'].append(float(t_cbo))
            data[dimension]['pso'].append(float(t_pso))

    for dimension, algos in data.items():
        # Dump Softmin (Decay)
        with open(f"results/SoftMin_decay_ackley_function_dim{dimension}.pkl", 'wb') as f:
            pkl.dump(np.array(algos['softmin_decay']), f)
            
        # Dump Simulated Annealing
        with open(f"results/SimulatedAnnealing_ackley_function_dim{dimension}.pkl", 'wb') as f:
            pkl.dump(np.array(algos['sa']), f)
        
        # Dump CBO
        with open(f"results/CBO_ackley_function_dim{dimension}.pkl", 'wb') as f:
            pkl.dump(np.array(algos['cbo']), f)
            
        # Dump PSO
        with open(f"results/PSO_ackley_function_dim{dimension}.pkl", 'wb') as f:
            pkl.dump(np.array(algos['pso']), f)
            
    logger.info("Pickle reconstruction complete.")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    # Generate the flattened workload
    all_tasks = [(d, s) for d in range(10, 100, 10) for s in range(100)]

    # Cross-reference with the disk state to strip completed runs
    completed = get_completed_tasks()
    pending_tasks = [t for t in all_tasks if t not in completed]
    
    logger.info(f"High-Dim Ackley | Total: {len(all_tasks)} | Done: {len(completed)} | Pending: {len(pending_tasks)}")

    # CSV State-Management Setup
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['dimension', 'seed', 'softmin_decay', 'sa', 'cbo', 'pso'])

        if pending_tasks:
            # Query Slurm variables to prevent core oversubscription
            N_CPU = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
            logger.info(f"Deploying worker pool across {N_CPU} threads...")
            
            with mp.Pool(N_CPU) as pool:
                # Process strictly unordered to maximize throughput
                for result in pool.imap_unordered(worker_task, pending_tasks):
                    writer.writerow(result)
                    
                    # Hard-sync to persistent storage
                    csvfile.flush()
                    os.fsync(csvfile.fileno())
                    
                    logger.info(f"Completed -> dimension: {result[0]}, Seed: {result[1]}")

    logger.info("High-dimensional simulations terminated normally.")
    reconstruct_pkl_files_from_csv()