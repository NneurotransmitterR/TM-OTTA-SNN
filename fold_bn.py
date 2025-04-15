import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from models.neurons import BNLIFNode


def fold_vbn(model: nn.Module, normalize_residual: bool = False, running_stats: bool = False, learnable_vth: bool = False):
    """
    Fold the MPBNs in the model: extract BN statistics and parameters.
    """
    if not check_model_with_mpbn(model):
        raise ValueError("The model does not have any MPBN to be folded.")
    print("Start folding MPBNs...")
    for name, m in model.named_modules():
        if isinstance(m, neuron.BaseNode):
            for name_c, child in m.named_children():
                if isinstance(child, (nn.LazyBatchNorm1d, nn.LazyBatchNorm2d)):
                    print(f"Folding {name}.{name_c}...")
                    setattr(m, name_c, nn.Identity())
                    setattr(m, 'fold_bn', True)
                    if learnable_vth:
                        with torch.no_grad():
                            m.vth.copy_((m.v_threshold - child.bias) * torch.sqrt(child.running_var + child.eps) / child.weight + child.running_mean)
                            print("The folded Vth is initialized.")
                    else:
                        setattr(m, 'normalize_residual', normalize_residual)
                        setattr(m, 'running_stats', running_stats)
                        setattr(m, 'mu', child.running_mean)
                        setattr(m, 'sigma2', child.running_var)
                        setattr(m, 'gamma', nn.Parameter(child.weight))
                        setattr(m, 'beta', nn.Parameter(child.bias))
                        setattr(m, 'eps', child.eps)

    print(f"Folding MPBNs done.")
    return model


def check_model_with_mpbn(model: nn.Module):
    """
    Check if the model can be adapted with TM, i.e., if it has MPBN.
    """
    for module in model.modules():
        if isinstance(module, neuron.BaseNode):
            if isinstance(getattr(module, 'vbn', nn.Identity()), (nn.LazyBatchNorm1d, nn.LazyBatchNorm2d)):
                return True
            else:
                continue
    return False
