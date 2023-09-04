from typing import Any
from collections import namedtuple

import torch
from torch.distributions import Transform, biject_to


HybridParts = namedtuple("HybridParts", "q_Y_given_X critic sampler")


def get_bijection(dist: torch.distributions.Distribution) -> Transform:
    """unconstrained to constrained"""
    return biject_to(dist.support).inv
