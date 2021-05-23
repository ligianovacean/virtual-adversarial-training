import torch
from torch import nn
import torch.nn.functional as F


class VATLoss(nn.Module):

    def __init__(self, epsilon=1.0, iterations=2, xi=10.0):
        super().__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.xi = xi

    def _l2_normalize(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def generate_vap(self, X, network: nn.Module, prediction, device):
        d = torch.rand(X.shape).sub(0.5)
        d = self._l2_normalize(d)

        # d <- unitvector(H*d)
        for _ in range(self.iterations):
            r = d * self.xi
            r = r.to(device)
            r.requires_grad_()
            perturbation_prediction = network(X + r)
            perturbation_prediction = F.log_softmax(perturbation_prediction, dim=1)
            kl_div = F.kl_div(perturbation_prediction, prediction, reduction='batchmean')
            kl_div.backward()
            gradients = r.grad
            d = self._l2_normalize(gradients)
            network.zero_grad()

        return self.epsilon * d

    def forward(self, X, network: nn.Module, device):
        with torch.no_grad():
            prediction = network(X)
            prediction = F.softmax(prediction, dim=1)

        r_vadv = self.generate_vap(X, network, prediction, device)
        prediction_rvadv = network(X + r_vadv)
        prediction_rvadv = F.log_softmax(prediction_rvadv, dim=1)
        lds = F.kl_div(prediction_rvadv, prediction, reduction='batchmean')

        return lds
