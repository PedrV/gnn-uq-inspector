import os
import random
import pickle

import numpy as np
import torch

from numpy.random import SeedSequence


def generate_experiment_seeds(num_experiments, entropy_path=None, master_seed=None):
    """
    Generates a list of statistically independent seeds for multiple experiments.
    If entropy_path and master_seed are None, generate and save entropy from OS state.

    Args:
        num_experiments (int): The number of independent seeds to generate.
        entropy_path (str/path like) Optional: location of pickled entropy.
        master_seed (int) Optional: The single, known seed for overall reproducibility.

    Returns:
        list: A list of integers, where each integer is a high-quality seed.
    """
    # SeedSequence uses hashing to ensure the master seed maps to a high-quality state.
    _does_not_exist_but_wanted = False
    if entropy_path is not None:
        if os.path.exists(entropy_path):
            print("Seeder: Restoring previous entropy ...")
            with open(entropy_path, "rb") as file:
                master_seed = pickle.load(file)
        else:
            _does_not_exist_but_wanted = True
            master_seed = None

    if (entropy_path is None or _does_not_exist_but_wanted) and master_seed is None:
        print("Seeder: Creating new entropy ...")

        directory = os.path.dirname(entropy_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        ss = SeedSequence()
        with open(entropy_path, "wb") as file:
            pickle.dump(ss.entropy, file)
    else:
        print("Seeder: Restoring previous entropy from master seed ...")
        ss = SeedSequence(master_seed)

    # Spawn N statistically independent child seeds.
    # The spawning process guarantees the child seeds are very far apart in the
    # pseudo-random number state space
    child_seeds = ss.spawn(num_experiments)
    return [s.generate_state(1)[0] for s in child_seeds]


def seed_everything(seed):
    seed = int(seed)
    # https://discuss.pytorch.org/t/does-pytorch-change-its-internal-seed-during-training/46505<
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # If you want to use deterministic algorithms with CUDA, then you need to set
    # the CUBLAS_WORKSPACE_CONFIG environment variable; otherwise, Torch errors.
    # See https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed, device):
    g = torch.Generator(device)
    g.manual_seed(seed)
    return g
