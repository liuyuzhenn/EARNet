import time

import torch
from tqdm import tqdm

from ..rotation_utils import *


def get_neibor_graph(rel_rot_inds, n, weight=None):
    if weight is None:
        matrix = torch.zeros((n, n), dtype=torch.uint8)
        for inds in rel_rot_inds:
            idx1, idx2 = int(inds[0]), int(inds[1])
            matrix[idx1, idx2] = 1
            matrix[idx2, idx1] = 1
    else:
        b = weight.shape[0]
        matrix = torch.zeros((b, n, n), dtype=torch.float32)
        for idx, inds in enumerate(rel_rot_inds):
            idx1, idx2 = int(inds[0]), int(inds[1])
            matrix[:, idx1, idx2] = weight[:, idx]
            matrix[:, idx2, idx1] = weight[:, idx]

    return matrix


def get_spanning_tree(graph, root_id=None):
    if root_id is None:
        root_id = int(torch.argmax(torch.sum(graph, dim=1)))
    n = graph.shape[0]
    tree = {}
    queue = [root_id]
    matrix_copy = torch.clone(graph)
    matrix_copy[:, root_id] = 0
    while len(queue) != 0:
        cur = queue[0]
        tree[cur] = [i for i in range(n) if matrix_copy[cur, i] == 1]
        del queue[0]
        queue += tree[cur]
        matrix_copy[:, tree[cur]] = 0
    return root_id, tree


def get_rotations_by_inds(rel_rots, rel_rot_inds):
    rots = {}
    for i, inds in enumerate(rel_rot_inds):
        id1, id2 = inds
        key1 = f"{id1}_{id2}"
        key2 = f"{id2}_{id1}"
        # rots[key1] = rel_rots[i]
        # rots[key2] = quaternion_invert(rel_rots[i])
        if key1 not in rots.keys():
            rots[key1] = rel_rots[i]
            rots[key2] = quaternion_invert(rel_rots[i])
        else:
            q1 = rots[key1]
            q = (q1 + rel_rots[i]) / 2
            q = F.normalize(q, p=2, dim=-1)
            rots[key1] = q
            rots[key2] = quaternion_invert(q)
    return rots


class CARA:
    def __init__(self, num_iters=5, cost_fun="L2", **kwargs) -> None:
        self.iters = num_iters
        self.cost_fun = cost_fun
        self.kwargs = kwargs
        self.threshold = kwargs.get("threshold", -1)

    def solve(
        self, rel_rots, rel_rot_inds, root_id=0, weight=None, use_CAI=False, threshold=0
    ):
        """
        rel_rots: N,3,3 (or ...,N,3,3)
        rel_rot_inds: E,2
        weight: ...,E
        """
        dims = rel_rots.shape[: rel_rots.dim() - 3]
        nodes = rel_rot_inds.max() + 1
        edges = rel_rot_inds.shape[0]

        device = rel_rots.device
        rel_mat = rel_rots
        rel_rots = self.inds2rotation(rel_rots, rel_rot_inds)
        if not use_CAI:
            graph = get_neibor_graph(rel_rot_inds, nodes)
            _, tree = get_spanning_tree(graph, 0)
            rots = self.initialize(0, tree, rel_rots, nodes, device)  # n,3,3
        else:
            assert weight is not None
            graph = get_neibor_graph(rel_rot_inds, nodes, weight)
            rots = self.CAI(0, graph, rel_rots, device)
            weight[weight < threshold] = 1e-6

        if weight is not None:
            # weight = weight/torch.sum(weight,dim=-1,keepdim=True)
            weight = (
                (torch.stack((weight, weight, weight), dim=len(dims)).transpose(-1, -2))
                .flatten(-2)
                .unsqueeze(-2)
            )

        I = torch.eye(3, dtype=torch.float32, device=device)
        A = torch.zeros(
            (*dims, edges * 3, (nodes - 1) * 3), dtype=torch.float32, device=device
        )
        it = 0
        for it in range(self.iters):
            b = torch.zeros((*dims, edges * 3, 1), dtype=torch.float32, device=device)
            rel_w = (
                rots[..., rel_rot_inds[:, 1], :, :].transpose(-1, -2)
                @ rel_mat
                @ rots[..., rel_rot_inds[:, 0], :, :]
            )
            rel_w = matrix_to_axis_angle(rel_w)
            for e in range(len(rel_rot_inds)):
                i = rel_rot_inds[e, 0]
                j = rel_rot_inds[e, 1]
                w = rel_w[..., e, :]

                # A is fixed during iteration
                if it == 0:
                    if i != 0:
                        A[..., e * 3 : (e + 1) * 3, (i - 1) * 3 : i * 3] = -I
                    if j != 0:
                        A[..., e * 3 : (e + 1) * 3, (j - 1) * 3 : j * 3] = I

                b[..., e * 3 : (e + 1) * 3, 0] = w

            if it == 0:
                if weight is not None:
                    AtW = A.transpose(-1, -2) * weight
                else:
                    AtW = A.transpose(-1, -2)

            if self.cost_fun == "GM":
                sig = self.kwargs["sigma"]
                weight_ = (
                    1 / (b.reshape(*dims, edges, 3).pow(2).sum(-1) + sig**2) ** 2
                )
                weight_ = (
                    (
                        torch.stack(
                            (weight_, weight_, weight_), dim=len(dims)
                        ).transpose(-2, -1)
                    )
                    .flatten(-2)
                    .unsqueeze(-2)
                )
                AtW_ = AtW * weight_
            elif self.cost_fun == "L0.5":
                # WARN: Training is unstable, which leads to NAN loss
                weight_ = b.reshape(*dims, edges, 3).pow(2).sum(-1).pow(3 / 8) + 1e-4
                weight_ = 1 / weight_
                weight_ = (
                    (
                        torch.stack(
                            (weight_, weight_, weight_), dim=len(dims)
                        ).transpose(-2, -1)
                    )
                    .flatten(-2)
                    .unsqueeze(-2)
                )
                AtW_ = AtW * weight_
            elif self.cost_fun == "Cauchy":
                sig = self.kwargs["sigma"]
                weight_ = b.reshape(*dims, edges, 3).pow(2).sum(-1) / sig**2 + 1
                weight_ = 1 / weight_
                weight_ = (
                    (
                        torch.stack(
                            (weight_, weight_, weight_), dim=len(dims)
                        ).transpose(-2, -1)
                    )
                    .flatten(-2)
                    .unsqueeze(-2)
                )
                AtW_ = AtW * weight_
            elif self.cost_fun == "L2":
                AtW_ = AtW
            else:
                raise NotImplementedError

            try:  # may fail due to singular matrix error
                if self.cost_fun == "L2":
                    if it == 0:
                        inv = torch.linalg.inv(AtW_ @ A)  # (A^tPA)^-1
                else:
                    inv = torch.linalg.inv(AtW_ @ A)
            except:
                return None, None

            dx = (inv @ (AtW_ @ b)).reshape(*dims, -1, 3)  # (n-1)*3

            if self.threshold > 0:
                n = torch.norm(dx, p=2, dim=-1).max()
                if n <= self.threshold:
                    break
            dx = axis_angle_to_matrix(dx)
            rots[..., 1:, :, :] = rots[..., 1:, :, :].clone() @ dx

        return rots, it + 1

    def inds2rotation(self, rel_rots, rel_rot_inds):
        rots = {}
        for i, inds in enumerate(rel_rot_inds):
            id1, id2 = inds
            key1 = f"{id1}_{id2}"
            key2 = f"{id2}_{id1}"
            R = rel_rots[..., i, :, :]
            rots[key1] = R
            rots[key2] = R.transpose(-1, -2)
        return rots

    def get_weight_matrix(self, weight, inds):
        n = torch.max(inds) + 1
        b = weight.shape[0]
        m = torch.zeros((b, n, n), dtype=torch.float32, device=weight.device)
        for idx, (id1, id2) in enumerate(inds):
            m[:, id1, id2] = weight[:, idx]
            m[:, id2, id1] = weight[:, idx]
        return m

    # TODO: This implementation is very slow for large graphs
    def CAI(self, root_id, graph, rel_rots, device):
        """Prim's maximun spanning tree algorithm 
        graph:  B,N,N
        rel_rots: dict: i_j->3,3
        """
        B, N = graph.shape[:2]
        rot_pred = torch.zeros((B, N, 3, 3), dtype=torch.float32, device=device)
        for b in range(B):
            G = graph[b]
            root = torch.argmax(G.sum(dim=0), dim=0)
            selected = [False for _ in range(N)]
            selected[root] = True
            rot_pred[b, root] = torch.eye(3, dtype=torch.float32, device=device)
            counter = 0
            edges = []
            while counter < N - 1:
                maximum = -1
                x = 0
                y = 1
                for i in range(N):
                    if selected[i]:
                        for j in range(N):
                            if not selected[j]:
                                if maximum < G[i, j]:
                                    maximum = G[i, j]
                                    x = i
                                    y = j
                edges.append([x, y])
                selected[y] = True
                counter += 1
                rot_pred[b, y] = rel_rots[f"{x}_{y}"][b] @ rot_pred[b, x].clone()

            rot_pred[b] = (
                rot_pred[b].clone()
                @ rot_pred[b, root_id : root_id + 1].transpose(-1, -2).clone()
            )
        return rot_pred

    def initialize(self, root_id, tree, rel_rots, n, device):
        val = list(rel_rots.values())[0]
        dim = val.dim()
        rot_pred = torch.zeros(
            (*val.shape[: dim - 2], n, 3, 3), dtype=torch.float32, device=device
        )
        rot_pred[..., root_id, :, :] = torch.eye(3, dtype=torch.float32, device=device)

        # first in first out
        queue = [root_id]
        while len(queue) != 0:
            ind_cur = queue[0]
            R_cur = rot_pred[..., ind_cur, :, :].clone()
            for ind in tree[ind_cur]:
                rot_pred[..., ind, :, :] = rel_rots[f"{ind_cur}_{ind}"] @ R_cur
            del queue[0]
            queue += tree[ind_cur]
        return rot_pred
