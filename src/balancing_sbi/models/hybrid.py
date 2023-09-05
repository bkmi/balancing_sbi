from functools import partial, wraps
import importlib
import os
import torch
import torch.nn as nn

from misbi.mist.surrogate import HybridSurrogate

from balancing_sbi.models.base import Model, ModelFactory
from balancing_sbi.benchmarks._extras import HybridParts


def flip(func):
    'Create a new function from the original with the arguments reversed'
    @wraps(func)
    def newfunc(*args):
        return func(*args[::-1])
    return newfunc


class HybridFactory(ModelFactory):
    def __init__(self, config, benchmark, simulation_budget):
        super().__init__(config, benchmark, simulation_budget, HybridModel)

    def get_train_time(self, benchmark_time, epochs):
        return 2*super().get_train_time(benchmark_time, epochs)

class HybridWithEmbedding(nn.Module):
    def __init__(self, hybrid, embedding):
        super().__init__()
        self.hybrid: HybridSurrogate = hybrid
        self.embedding: torch.nn.Module = embedding

    def forward(self, theta, x):
        """this method also makes our model obey the lampe (theta, x) argument ordering"""
        return self.hybrid.approx_unnormalized_log_p_y_given_x(self.embedding(x), theta)

    def sample(self, x, shape):
        return self.hybrid.sample(self.embedding(x), shape, show_progress_bars=False)
    
    def loss(self, theta, x):
        """this method also makes our model obey the lampe (theta, x) argument ordering"""
        return self.hybrid.loss(x, theta)

class HybridModel(Model):
    model_file_name = "model.pt"
    embedding_file_name = "embedding.pt"

    def __init__(self, benchmark, model_path, config):

        self.observable_shape = benchmark.get_observable_shape()
        self.embedding_dim = benchmark.get_embedding_dim()
        self.parameter_dim = benchmark.get_parameter_dim()
        self.device = benchmark.get_device()

        self.model_path = model_path

        self.prior = benchmark.get_prior()

        embedding_build = benchmark.get_embedding_build()
        self.embedding = embedding_build(self.embedding_dim, self.observable_shape).to(self.device)

        # density_surrogate_fn = self._get_class_init_fn(config["density_surrogate"])
        # density_surrogate = density_surrogate_fn(
        #     features=self.parameter_dim,
        #     context=self.embedding_dim,
        #     **config["density_surrogate_kwargs"],
        # )
        # ratio_surrogate_fn = self._get_class_init_fn(config["ratio_surrogate"])
        # ratio_surrogate = ratio_surrogate_fn(
        #     x_dim=self.embedding_dim,
        #     y_dim=self.parameter_dim,
        #     **config["ratio_surrogate_kwargs"],
        # )
        # sampler = partial(
        #     RejectionSampler, 
        #     transform_y_c_to_y_u=None, # ${get_transform:${task.name},${task.benchmark}} TODO
        #     **config["sampler_kwargs"], 
        # )

        hybrid_parts: HybridParts = benchmark.get_hybrid_build()
        q_Y_given_X = hybrid_parts.q_Y_given_X(
            features=self.parameter_dim,
            context=self.embedding_dim,
        )
        density_surrogate_fn = self._get_class_init_fn(config["density_surrogate"], 2)
        density_surrogate = density_surrogate_fn(
            q_Y_given_X=q_Y_given_X,
            neg_samples=None,  # not necessary here
        )
        critic = hybrid_parts.critic(
            x_dim=self.embedding_dim,
            y_dim=self.parameter_dim,
        )
        ratio_surrogate_fn = self._get_class_init_fn(config["ratio_surrogate"])
        ratio_surrogate = ratio_surrogate_fn(
            critic=critic,
            sampler=None,  # not necessary here
            p_Y_0=self.prior,
            neg_samples=config["neg_samples"],
        )
        self.hybrid = HybridSurrogate(
            density_surrogate=density_surrogate,
            ratio_surrogate=ratio_surrogate,
            sampler=hybrid_parts.sampler,
        )
        self.model = HybridWithEmbedding(self.hybrid, self.embedding)

    @staticmethod
    def _get_class_init_fn(class_path, maxsplit: int = 1):
        module_name, *class_name = class_path.rsplit(".", maxsplit=maxsplit)
        module = importlib.import_module(module_name)
        if isinstance(class_name, list):
            submodule = module
            for cn in class_name:
                submodule = getattr(submodule, cn)
            return submodule
        else:
            return getattr(module, class_name)

    @classmethod
    def is_trained(cls, model_path):
        return (os.path.exists(os.path.join(model_path, cls.embedding_file_name)) and os.path.exists(os.path.join(model_path, cls.model_file_name)))

    def get_loss_fct(self, config):
        def wrapped_loss(model):
            return self.model.loss
        return wrapped_loss

    def log_prob(self, theta, x):
        x = x.to(self.device)
        theta = theta.to(self.device)
        return self.model(theta, x)

    def get_posterior_fct(self):
        def get_posterior(x):
            class Posterior():
                def __init__(self, sampling_fct, log_prob_fct):
                    self.sample = sampling_fct
                    self.log_prob = log_prob_fct

            return Posterior(
                lambda shape: self.model.sample(x.to(self.device), shape).cpu(), 
                lambda theta: self.model(theta.to(self.device), x.to(self.device)).cpu(),
            )

        return get_posterior

    def __call__(self, theta, x):
        return self.log_prob(theta, x)

    def sampling_enabled(self):
        return False

    def save(self):
        torch.save(self.embedding.state_dict(), os.path.join(self.model_path, self.embedding_file_name))
        torch.save(self.model.state_dict(), os.path.join(self.model_path, self.model_file_name))

    def load(self):
        self.embedding.load_state_dict(torch.load(os.path.join(self.model_path, self.embedding_file_name)))
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, self.model_file_name)))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
