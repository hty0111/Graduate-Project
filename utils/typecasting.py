"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-02-22 12-26
Description: 
"""

import torch
import numpy as np
import gymnasium as gym

def n2t(data: np.ndarray, **kwargs) -> torch.Tensor:
    data = torch.from_numpy(data) if type(data) == np.ndarray else data
    data = data.to(**kwargs) if kwargs is not None else data
    return data

def t2n(x: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

def space_dict_to_space(x: dict) -> gym.spaces.Space:
    return list(x.values())[0]

def space_to_shape(x) -> tuple:
    if x.__class__.__name__ == 'Discrete':
        return x.n,
    elif x.__class__.__name__ == "Box":
        return x.shape
    elif x.__class__.__name__ == 'list':
        return len(x),
    elif x.__class__.__name__ == 'ndarray':
        return x.shape

def shape_to_dim(x: tuple) -> int:
    return x[0]

def space_dict_to_shape(x: dict) -> tuple:
    return space_to_shape(space_dict_to_space(x))

def space_to_dim(x) -> int:
    return shape_to_dim(space_to_shape(x))

def space_dict_to_dim(x: dict) -> int:
    return shape_to_dim(space_dict_to_shape(x))

def dict_to_value_list(x: dict) -> list:
    return list(x.values())


if __name__ == "__main__":
    obs = np.array([0, 1])
    print(space_to_shape(obs))

