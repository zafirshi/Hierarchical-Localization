import torch
import numpy as np
from hloc.utils.base_model import BaseModel


def find_nn(dist, sim, ratio_thresh, distance_thresh):
    dist_nn, ind_nn = torch.topk(-1 * dist, 2 if ratio_thresh else 1, dim=-1, largest=True, sorted=True)
    dist_nn = -1 * dist_nn

    sim_nn, _, = torch.topk(sim, 2 if ratio_thresh else 1, dim=-1, largest=True, sorted=True)
    assert np.array_equal(ind_nn.cpu().numpy(), _.cpu().numpy())  # idx for min_dist should be same with idx for max_sim

    # Importance: (min, sub_min) order
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=dist.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= ratio_thresh * dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    # not calculate (sim_nn[..., 0]+1)/2 cause dist_nn != 2 * (1 - sim_nn)
    scores = torch.where(mask, sim_nn[..., 0], sim_nn.new_tensor(0))
    return matches, scores


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighbor(BaseModel):
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        pass

    def _forward(self, data):
        for key in self.required_inputs:
            assert key in data, 'Missing key {} in data'.format(key)
        if data['descriptors0'].size(-1) == 0 or data['descriptors1'].size(-1) == 0:
            matches0 = torch.full(
                data['descriptors0'].shape[:2], -1,
                device=data['descriptors0'].device)
            return {
                'matches0': matches0,
                'matching_scores0': torch.zeros_like(matches0)
            }
        ratio_threshold = self.conf['ratio_threshold']
        if data['descriptors0'].size(-1) == 1 or data['descriptors1'].size(-1) == 1:
            ratio_threshold = None

        d0, d1 = data['descriptors0'], data['descriptors1']
        dist = torch.einsum('bdn,bdm->bnm', 1 - d0, d1) + torch.einsum('bdn,bdm->bnm', d0, 1 - d1)
        sim = torch.einsum('bdn,bdm->bnm', d0, d1)
        matches0, scores0 = find_nn(dist, sim, ratio_threshold, self.conf['distance_threshold'])

        if self.conf['do_mutual_check']:
            matches1, scores1 = find_nn(dist.transpose(1, 2), sim.transpose(1, 2),
                                        ratio_threshold, self.conf['distance_threshold'])
            matches0 = mutual_check(matches0, matches1)

        return {
            'matches0': matches0,
            'matching_scores0': scores0,
        }
