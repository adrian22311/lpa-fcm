import numpy as np
from itertools import permutations, combinations
import time

### Deng, Zheng-Hong, et al. "A complex network community detection 
### algorithm based on label propagation and fuzzy C-means." 
### Physica A: Statistical Mechanics and its Applications 519 (2019): 217-226.


# (1) The establishment of order access sequence. The neighbor evaluation vectors are calculated by feature vectors. Order access sequence is generated base on size of neighbor evaluation vertexes.
# (2) The assignment of initial community labels. The initialization process is divided into two steps. (a) The formation of initial communities. (b) The fusion of isolated vertexes. Labels of vertexes are assigned base on their affiliated community labels.
# (3) The modification of vertexes labels. Some unstable vertexes are selected to change their community labels by Fuzzy C-mean neighbor membership vector.
# (4) Parameter update. The KL-FCM parameters are updated until cut-off conditions are satisfied during the process of iterative calculation.
def neighbor_eval(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    neighbor_eval_score = np.zeros((num_nodes, num_nodes))

    M_avg = np.sum(adj_matrix, axis=0) / num_nodes
    beta = np.sum(np.linalg.norm(adj_matrix - M_avg, axis=1))  # / num_nodes

    for i in range(num_nodes):
        M_i = adj_matrix[i]
        # Calculate neighbor evaluation between vertex V_i and its neighbors
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                M_j = adj_matrix[j]
                neighbor_eval_score[i, j] = np.exp(-np.linalg.norm(M_i - M_j) / beta)

    return neighbor_eval_score

def order_access_sequence(neighbor_evaluation: np.ndarray) -> np.ndarray:
    return np.argsort(neighbor_evaluation.sum(axis=1))[::-1]  # descending order

# The neighbor evaluation function is usually a decreasing function. The larger function value is, the closer relationship between vertex and its neighbor vertex is. Vertex and its closely neighbor vertexes have larger probability to share the same community label. The elements of vector f[i]
#  are sorted in descending order. The vertexes of larger elements in vector f[i]
#  are assigned into vertex V[i]’s neighbor evaluation set.
def calculation_eval_set(neighbor_evaluation: np.ndarray) -> dict:
    neighbor_set = {}
    for i in range(len(neighbor_evaluation)):
        neighbor_set[i] = np.argsort(
            neighbor_evaluation[i][neighbor_evaluation[i] != 0]
        )[::-1]
    return neighbor_set



def sum_neighbor_set_eval(
    neighbor_set: dict, neighbor_evaluation: np.ndarray, i: int, j: int | None = None
) -> float:
    score = 0.0
    if j is None:
        iterator = permutations(neighbor_set[i], 2)
    else:
        iterator = zip(neighbor_set[i], neighbor_set[j])
    for i, j in iterator:
        score += np.linalg.norm(neighbor_evaluation[i] - neighbor_evaluation[j])
    return score

def reverse_non_unique_mapping(d):
    dinv = {}
    for k, v in d.items():
        if v in dinv:
            dinv[v].append(k)
        else:
            dinv[v] = [k]
    return dinv

def average_distance_to_community(
    k: int,
    c: int,
    communities: dict,
    # neighbor_set: dict,
    neighbor_evaluation: np.ndarray,
) -> float:
    N_c = communities[c]
    lN_c = len(N_c) if len(N_c) > 1 else 2
    _sum = 0.0
    for i in N_c:
        _sum += np.linalg.norm(neighbor_evaluation[k] - neighbor_evaluation[i])

    return 2 / (lN_c * (lN_c - 1)) * _sum



def initial_communities(
    # adj_matrix: np.ndarray,
    neighbor_evaluation: np.ndarray,
    neighbor_set: dict,
    threshold: float = 1e-3,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    f = neighbor_evaluation
    Nf = neighbor_set
    communities = {}
    order_access = order_access_sequence(f)

    _sums = np.zeros((len(order_access), len(order_access)))
    __c = 0
    for c, i in enumerate(order_access):
        if communities.get(i) is None:
            communities[i] = __c
            __c += 1
        _sums[i][i] = sum_neighbor_set_eval(Nf, f, i)

        for j in order_access:
            if i == j:
                continue
            _sums[i][j] = sum_neighbor_set_eval(Nf, f, i, j)
            lNf_i = len(Nf[i])
            lNf_j = len(Nf[j])
            lNf_i = lNf_i if lNf_i > 1 else 2
            lNf_j = lNf_j if lNf_j > 1 else 2

            d_i = 2 / (lNf_i * (lNf_j - 1)) * _sums[i][i]
            d_j = 2 / (lNf_j * (lNf_i - 1)) * _sums[j][j]
            d_ij = 1 / (lNf_i * lNf_j) * _sums[i][j]

            if np.abs(d_ij - (d_i + d_j) / 2) <= threshold:
                communities[j] = communities[i]

    # form K base communities
    communities_inv = reverse_non_unique_mapping(communities)
    cs = np.unique(list(communities.values()))
    for k, c in communities.items():
        min_sc = np.inf
        min_c = []
        for _c in cs:
            sc = average_distance_to_community(k, _c, communities_inv, f)
            if sc <= min_sc:
                min_sc = sc
                min_c.append(_c)
        if c not in min_c:
            for _min_c in min_c:
                if k not in communities_inv[_min_c]:
                    communities_inv[_min_c].append(k)
    return communities_inv, communities


# modify unstable vertexes
def modified_unstable_vertexes(
    neighbor_evaluation: np.ndarray,
    neighbor_set: dict,
    communities_inv: dict[
        int, list[int]
    ],  # community: list of vertexes (with possible vertexes)
    communities: dict[int, int],  # vertex: initial community
) -> tuple[dict[int, int], np.ndarray]:
    f = neighbor_evaluation
    Nf = neighbor_set
    # establish membership vector
    K = len(communities_inv)
    u = np.zeros((len(f), K), np.float32)
    for i, v in enumerate(f):
        _s2 = np.sum(v[neighbor_set[i]]) + 1e-6
        for c in range(K):
            _s1 = np.sum(v[np.intersect1d(Nf[i], communities_inv[c])])

            u[i][c] = _s1 / _s2

    # selection of unstable vertexes
    unstable_vertexes = {}

    for i, v in enumerate(u):
        j = communities[i]
        # p = np.max(v)
        ps = (v == v.max()).nonzero()[0]
        if v.sum() > 0 and j not in ps:
            unstable_vertexes[i] = np.random.choice(ps)
            communities[i] = unstable_vertexes[i]

    return communities, u


def objective_function(u, fd, pi):
    s = 0.0
    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            s += u[i][j] * fd[i][j] + u[i][j] * np.log(
                u[i][j] + 1e-3 / (pi[i][j] + 1e-3)
            )
    return s


def parameters_update(
    adj_matrix: np.ndarray,
    neighbor_evaluation: np.ndarray,
    neighbor_set: dict,
    communities_inv: dict[
        int, list[int]
    ],  # community: list of vertexes (with possible vertexes)
    u: np.ndarray,
    mi: None | np.ndarray = None,
    delta: None | np.ndarray = None,
    pi: None | np.ndarray = None,
    fd: None | np.ndarray = None,
):
    f = neighbor_evaluation
    Nf = neighbor_set
    K = u.shape[1]
    mi = mi or np.zeros((K))
    delta = delta or np.zeros((K))
    pi = pi or np.zeros((K, len(f)))
    fd = fd or np.zeros((K, len(f))).tolist()
    for i in range(K):
        mi[i] = np.mean(f[communities_inv[i]])
        delta[i] = np.std(f[communities_inv[i]])

        for j in range(f.shape[0]):
            NbVjCi = np.intersect1d(Nf[j], communities_inv[i])
            if len(NbVjCi) == 0:
                pi[i][j] = 0
            else:
                pi[i][j] = np.sum(u[NbVjCi, i]) / len(NbVjCi)
            fd[i][j] = (
                1 / 2 * np.log(2 * np.pi)
                + np.log(delta[i])
                + (adj_matrix[j] - mi[i]) ** 2 / (2 * (delta[i] ** 2))
            )
    return mi, delta, pi, fd


def map_ids_by_value(communities: dict[int, int]) -> dict[int, int]:
    uniq_val = np.unique(list(communities.values()))
    _mapper = dict(zip(uniq_val, np.arange(0, len(uniq_val) + 0)))
    return {k: _mapper[v] for k, v in communities.items()}


def map_ids_by_key(communities: dict[int, int]) -> dict[int, int]:
    uniq_val = list(communities.keys())
    _mapper = dict(zip(uniq_val, np.arange(0, len(uniq_val) + 0)))
    return {_mapper[k]: v for k, v in communities.items()}


def lpa_fcm(adj_matrix, threshold=0.5, gamma=1e-6, max_iter=100) -> tuple:
    start = time.perf_counter()
    f = neighbor_eval(adj_matrix)
    Nf = calculation_eval_set(f)
    comm_init_inv, comm_init = initial_communities(f, Nf, threshold=threshold)

    iter = 0
    prev = None
    while True:
        comm_init_inv = map_ids_by_key(comm_init_inv)
        comm_init, u = modified_unstable_vertexes(f, Nf, comm_init_inv, comm_init)

        mi, delta, pi, fd = parameters_update(adj_matrix, f, Nf, comm_init_inv, u)

        if iter == 0:
            prev = np.zeros((u.shape[0], len(comm_init)))
        obj = objective_function(u.T, fd, pi)
        if (obj := np.linalg.norm(prev - obj)) <= gamma:
            break
        iter += 1
        if iter >= max_iter:
            break
    comm = map_ids_by_value(comm_init)
    return comm, time.perf_counter() - start
