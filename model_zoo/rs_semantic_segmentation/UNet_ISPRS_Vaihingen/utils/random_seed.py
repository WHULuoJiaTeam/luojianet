import random
import numpy as np
from luojianet_ms import set_seed

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
