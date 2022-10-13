import numpy as np
import torch

m=torch.Tensor(np.random.randint(1,5,(2,3,4)))
print(m.size(-1))