import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional
from models.neurons import BNLIFNode
from fold_bn import fold_vbn


__all__ = ['TM', 'check_model', 'configure_model', 'collect_params']
# Inspired by SAR (https://github.com/mr-eggplant/SAR/blob/main/sar.py)
class TM(nn.Module):
    """
    SNN with TM-ENT adaptation capability. Once TMed, a model adapts itself by updating on every forward pass.
    """
    def __init__(self, model, optimizer, steps=1, margin_e0=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "TM requires >= 1 step(s) to forward and update"
        self.margin_e0 = margin_e0

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer, margin=self.margin_e0)
        return outputs
    
    @staticmethod
    @torch.enable_grad()
    def forward_and_adapt(x: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, margin=None):
        optimizer.zero_grad()
        # we use single step forward, but assume all T steps are done in the model's forward()
        outputs = model(x)
        mean_output = outputs.mean(0)
        entropys = softmax_entropy(mean_output)
        if margin is not None:
            filter_id_1 = torch.where(entropys < margin)
            entropys = entropys[filter_id_1]
        loss = entropys.mean(0)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
        return outputs


@torch.jit.script
def softmax_entropy(x: torch.Tensor):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def check_model(model: nn.Module):
    """
    Check if the model can be adapted with TM.
    """
    assert model.training, "TM needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "TM needs params to update, check which require grad"
    assert not has_all_params, "TM should not update all params, check which require grad"
    has_mpbn = any([isinstance(m, (BNLIFNode,)) for m in model.modules()])
    assert has_mpbn, "TM needs MPBNs for its optimization"


def configure_model(model: nn.Module, fold_bn, normalize_residual, running_stats, learnable_vth=False):
    """
    Configure the model for TM adaptation, i.e., enable grads and stats for MPBNs.
    """
    model.train()
    model.requires_grad_(False)
    if fold_bn:
        model = fold_vbn(model, normalize_residual=normalize_residual, running_stats=running_stats,
                         learnable_vth=learnable_vth)
    for m in model.modules():
        if isinstance(m, (BNLIFNode,)):
            m.requires_grad_(True)
            if not fold_bn:
                for child in m.children():
                    if isinstance(child, (nn.LazyBatchNorm1d, nn.LazyBatchNorm2d)):
                        if not learnable_vth:
                            child.requires_grad_(True)
                        child.track_running_stats = False
                        child.running_mean = None
                        child.running_var = None
    return model


def collect_params(model: nn.Module, fold_bn, learnable_vth=False):
    """
    Collect the parameters and names to be optimized.
    """
    params = []
    names = []
    for name, m in model.named_modules():
        if isinstance(m, (BNLIFNode,)):
            if fold_bn:
                for np, p in m.named_parameters():
                    if 'gamma' in np or 'beta' in np or 'vth' in np:
                        params.append(p)
                        names.append(f'{name}.{np}')
            else:
                if learnable_vth:
                    for np, p in m.named_parameters():
                        if 'vth' in np or 'a' in np:
                            params.append(p)
                            names.append(f'{name}.{np}')
                else:
                    for nc, c in m.named_children():
                        if isinstance(c, (nn.LazyBatchNorm1d, nn.LazyBatchNorm2d)):
                            for np, p in m.named_parameters():
                                if 'vbn.weight' in np or 'vbn.bias' in np:
                                    params.append(p)
                                    names.append(f'{name}.{np}')
    return params, names
