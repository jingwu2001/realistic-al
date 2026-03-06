import random
import numpy as np
import torch
from contextlib import contextmanager

def save_random_states():
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return py_state, np_state, torch_state, cuda_state

def revert_to_random_state(state):
    py_state, np_state, torch_state, cuda_state = state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)

@contextmanager
def preserve_random_state():
    state = save_random_states()
    try:
        yield
    finally:
        revert_to_random_state(state)
