# Reproduced from https://github.com/montefiore-ai/hypothesis

from functools import partial
import torch
import zuko

from ._extras import HybridParts
from .base import Benchmark
from torch.distributions.multivariate_normal import MultivariateNormal as Normal

import misbi.mist.nn
import misbi.mist.sampler


class SLCP(Benchmark):
    def __init__(self):
        super().__init__("slcp")

        self._mu = [0.7, -2.9]
        self._p = torch.distributions.uniform.Uniform(-3.0, 3.0)

        self.LOWER = -3 * torch.ones(2).float()
        self.UPPER = 3 * torch.ones(2).float()
        
    @torch.no_grad()
    def get_prior(self):
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        return zuko.distributions.BoxUniform(self.LOWER, self.UPPER)

    @torch.no_grad()
    def simulate(self, parameters):
        """Perform a simulation

        Args:
            parameters (Tensor): The parameters conditioning the simulator

        Returns:
            Tensor: the synthetic observation
        """
        
        success = False

        while not success:
            try:
               if self._mu is None:
                   mean = torch.tensor([self._p.sample().item(), self._p.sample().item()]).float()
               else:
                   mean = torch.tensor(self._mu).float()
               scale = 1.0
               s_1 = parameters[0] ** 2
               s_2 = parameters[1] ** 2
               rho = self._p.sample().tanh()
               covariance = torch.tensor([
                   [scale * s_1 ** 2, scale * rho * s_1 * s_2],
                   [scale * rho * s_1 * s_2, scale * s_2 ** 2]])
               normal = Normal(mean, covariance)
               x_out = normal.sample((4,)).view(-1)
               success = True
            except ValueError:
                pass
        
        return x_out

    def get_simulation_batch_size(self):
        return 128

    def is_vectorized(self):
        return False

    def get_parameter_dim(self):
        return 2

    def get_observable_shape(self):
        return(8,)

    def get_classifier_build(self):
        return zuko.nn.MLP, {"hidden_features": [256]*6, "activation": torch.nn.SELU}

    def get_flow_build(self):
        return zuko.flows.NSF, {"hidden_features": [256]*3, "activation": torch.nn.SELU}

    def get_hybrid_build(self):
        q_Y_given_X = partial(
            misbi.mist.nn.MAF,
            transforms=5,
            randperm=True,
            hidden_features=[64],
            activation="relu",
        )
        critic = partial(
            misbi.mist.nn.JointCriticResMLP,
            hidden_dims=[128],
            activation="gelu",
            normalize=True,
        )
        sampler = partial(
            misbi.mist.sampler.RejectionSampler, 
            transform_y_c_to_y_u=self.get_transform_constrained_to_unconstrained(),
            max_sampling_batch_size=10_000,
            num_samples_to_find_max=10_000,
            num_iter_to_find_max=100,
            m=1.2,
            timeout_seconds=None,
        )
        return HybridParts(
            q_Y_given_X=q_Y_given_X,
            critic=critic,
            sampler=sampler,
        )

    def get_device(self):
        return "cpu"

    def get_domain(self):
        return self.LOWER, self.UPPER

    def get_nb_cov_samples(self):
        return 1000

    def get_cov_bins(self):
        return 100
    
    def get_simulate_nb_gpus(self):
        return 0

    def get_simulate_nb_cpus(self):
        return 1

    def get_simulate_ram(self, block_size):
        return 0.0001*block_size

    def get_simulate_time(self, block_size):
        return 10*block_size

    def get_merge_nb_gpus(self):
        return 0

    def get_merge_nb_cpus(self):
        return 1

    def get_merge_ram(self, dataset_size):
        return 0.0001*dataset_size

    def get_merge_time(self, dataset_size):
        return 0.1*dataset_size

    def get_train_nb_gpus(self):
        return 1

    def get_train_nb_cpus(self):
        return 1

    def get_train_ram(self):
        return 4

    def get_train_time(self, dataset_size):
        return 0.002*dataset_size + 60

    def get_test_nb_gpus(self):
        return 1

    def get_test_nb_cpus(self):
        return 1

    def get_test_ram(self):
        return 4

    def get_test_time(self, dataset_size):
        return 0.002*dataset_size + 600

    def get_coverage_nb_gpus(self):
        return 1

    def get_coverage_nb_cpus(self):
        return 1

    def get_coverage_ram(self):
        return 4

    def get_coverage_time(self, dataset_size):
        return 10 * dataset_size + 1800
