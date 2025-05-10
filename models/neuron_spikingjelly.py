# Built upon the original implementation and submitted to SpikingJelly.
# An alternative implementation of threshold re-parameterization is also included.
# If the PR is merged, you can import like: `from spikingjelly.activation_based.neuron import MPBNLIFNode`
import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import BaseNode
from spikingjelly.activation_based import surrogate, functional
from typing import Callable, Optional


class MPBNBaseNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode = 's', backend = 'torch', store_v_seq: bool = False, mpbn: bool = True, 
                 out_features = None, out_channels = None, learnable_vth: bool = False,
                 bn_momentum: float = 0.1, bn_decay_momentum: float = 0.94, bn_min_momentum: float = 0.005):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        assert out_features is None and out_channels is not None or out_features is not None and out_channels is None, \
            "One of out_features or out_channels should be specified."
        self.out_features = out_features
        self.out_channels = out_channels
        self.mpbn = mpbn
        if mpbn:
            if out_features is None and out_channels is not None:
                self.vbn = nn.LazyBatchNorm2d()
            else:
                self.vbn = nn.LazyBatchNorm1d()
        else:
            self.vbn = nn.Identity()

        self.register_buffer('mu', None)
        self.register_buffer('sigma2', None)
        self.gamma = None
        self.beta = None
        self.eps = None

        self.fold_bn = False
        self.normalize_residual = False
        self.running_stats = False

        self.bn_momentum = bn_momentum
        self.bn_decay_momentum = bn_decay_momentum
        self.bn_min_momentum = bn_min_momentum

        self.learnable_vth = learnable_vth
        if learnable_vth:  # train log(vth)
            self.a = nn.Parameter(torch.full((out_channels or out_features,), 0.))
        
        self.register_memory('vth', v_threshold)

    def init_vth(self, x: torch.Tensor):
        if isinstance(self.vth, float):
            if isinstance(self.v_threshold, float):
                self.vth = torch.full((x.shape[1],), self.v_threshold, device=x.device, dtype=x.dtype)
            else:
                self.vth = self.v_threshold
            self.vth_ = self.vth
    
    def compute_running_stats(self, v: torch.Tensor):
        if v.ndim == 2:
            if v.shape[0] == 1:  # do nothing for fc layer when bs=1
                return
            mu = torch.mean(v, dim=0).detach()
            sigma2 = torch.var(v, dim=0, unbiased=True).detach()
            if self.running_stats:
                if self.mu is None or self.sigma2 is None:
                    self.mu = mu
                    self.sigma2 = sigma2
                else:
                    self.mu = self.mu.detach() * (1 - self.bn_momentum) + mu * self.bn_momentum
                    self.sigma2 = self.sigma2.detach() * (1 - self.bn_momentum) + sigma2 * self.bn_momentum
                    self.bn_momentum = max(self.bn_momentum * self.bn_decay_momentum, self.bn_min_momentum)
            else:
                self.mu = mu
                self.sigma2 = sigma2
        elif v.ndim == 4:
            mu = torch.mean(v, dim=(0, 2, 3)).detach()
            sigma2 = torch.var(v, dim=(0, 2, 3), unbiased=True).detach()
            if self.running_stats:
                if self.mu is None or self.sigma2 is None:
                    self.mu = mu
                    self.sigma2 = sigma2
                else:
                    self.mu = self.mu.detach() * (1 - self.bn_momentum) + mu * self.bn_momentum
                    self.sigma2 = self.sigma2.detach() * (1 - self.bn_momentum) + sigma2 * self.bn_momentum
                    self.bn_momentum = max(self.bn_momentum * self.bn_decay_momentum, self.bn_min_momentum)
            else:
                self.mu = mu
                self.sigma2 = sigma2
        else:
            raise NotImplementedError(f"Only 2D and 4D tensor are supported, but got {v.ndim}D tensor.")
    
    def pre_charge(self, x: torch.Tensor):
        raise NotImplementedError("This method should be implemented in subclasses, e.g. the charging function of LIF neuron.")

    def neuronal_charge(self, x: torch.Tensor):
        self.pre_charge(x)
        self.v = self.vbn(self.v)
        if self.fold_bn and not self.learnable_vth and self.training:
            self.compute_running_stats(self.v)

    def neuronal_fire(self):
        if self.v.ndim == 2:
            if self.fold_bn and not self.learnable_vth:
                self.vth = (self.vth_ - self.beta) * torch.sqrt(self.sigma2 + self.eps) / self.gamma + self.mu
            if self.learnable_vth:
                self.vth = torch.exp(self.a)
            diff = self.v - self.vth.view(1, self.vth.shape[0])
            spike = self.surrogate_function(diff)
            if self.normalize_residual:
                mask = diff <= 0
                gamma = self.gamma.unsqueeze(0).expand_as(mask)
                mu = self.mu.unsqueeze(0).expand_as(mask)
                beta = self.beta.unsqueeze(0).expand_as(mask)
                sigma = torch.sqrt(self.sigma2 + self.eps).unsqueeze(0).expand_as(mask)
                normalized_residual = (self.v[mask] - mu[mask]) / sigma[mask] * gamma[mask] + beta[mask]
                self.v.masked_scatter_(mask, normalized_residual)
        elif self.v.ndim == 4:
            if self.fold_bn and not self.learnable_vth:
                self.vth = (self.vth_ - self.beta) * torch.sqrt(self.sigma2 + self.eps) / self.gamma + self.mu
            if self.learnable_vth:
                self.vth = torch.exp(self.a)
            diff = self.v - self.vth.view(1, self.vth.shape[0], 1, 1)
            spike = self.surrogate_function(diff)
            if self.normalize_residual:
                mask = diff <= 0
                gamma = self.gamma.view(1, -1, 1, 1).expand_as(mask)
                mu = self.mu.view(1, -1, 1, 1).expand_as(mask)
                beta = self.beta.view(1, -1, 1, 1).expand_as(mask)
                sigma = torch.sqrt(self.sigma2 + self.eps).view(1, -1, 1, 1).expand_as(mask)
                normalized_residual = (self.v[mask] - mu[mask]) / sigma[mask] * gamma[mask] + beta[mask]
                self.v.masked_scatter_(mask, normalized_residual)
        else:
            raise NotImplementedError(f"Only 2D and 4D tensors are supported, but got {self.v.ndim}D tensors.")
        return spike

    def single_step_forward(self, x: torch.Tensor):
        self.init_vth(x)
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def re_parameterize_v_threshold(self, normalize_residual: bool = False, running_stats: bool = False):
        # "re-parameterize" threshold to enable OTTA capability
        if isinstance(self.vbn, nn.Identity):
            if self.fold_bn == True:
                print(f"Re-parameterization has already been done in this neuron, skipping...")
            else:
                print(f"MPBN is not enabled in this neuron, skipping...")
            return
        self.fold_bn = True
        if self.learnable_vth:  # if self.a is learnable during training:
            with torch.no_grad():
                self.v_threshold = torch.exp(self.a)  # now e^a becomes the 'original' v_threshold, on which vth is calculated
            self.learnable_vth = False
        self.normalize_residual = normalize_residual
        self.running_stats = running_stats
        self.mu = self.vbn.running_mean
        self.sigma2 = self.vbn.running_var
        self.gamma = nn.Parameter(self.vbn.weight)
        self.beta = nn.Parameter(self.vbn.bias)
        self.eps = self.vbn.eps
        self.vbn = nn.Identity()


class MPBNLIFNode(MPBNBaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode = 's', backend = 'torch', store_v_seq: bool = False,
                 mpbn: bool = True, out_features = None, out_channels = None, learnable_vth: bool = False,
                 bn_momentum: float = 0.1, bn_decay_momentum: float = 0.94, bn_min_momentum: float = 0.005):
        
        assert isinstance(tau, float) and tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq,
                         mpbn, out_features, out_channels, learnable_vth, bn_momentum, bn_decay_momentum, bn_min_momentum)

        self.tau = tau
        self.decay_input = decay_input
    
    @property
    def supported_backends(self):
        return ('torch')

    def pre_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        return v


if __name__ == '__main__':
    neuron = MPBNLIFNode(mpbn=True, out_channels=3, learnable_vth=False, step_mode='m').cuda()
    neuron.a = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]).cuda())  # Example learnable parameter
    x_seq = torch.rand(4, 1, 3, 32, 32).cuda()  # Example input tensor
    spikes1 = neuron(x_seq)
    functional.reset_net(neuron)
    print(spikes1, spikes1.shape)

    # new way to re-parameterize threshold per neuron
    neuron.re_parameterize_v_threshold(running_stats=False, normalize_residual=True)

    spikes2 = neuron(x_seq)
    print(spikes2, spikes2.shape)
    print(torch.allclose(spikes2, spikes1))
    print((spikes1 == spikes2).float().sum() / spikes1.numel())

    print("Re-parameterization done.")