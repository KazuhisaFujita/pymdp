import jax.numpy as jnp

from functools import partial
from typing import Optional, Tuple, List
from jax import tree_util, nn, jit, vmap, lax
from jax.scipy.special import xlogy, digamma
from opt_einsum import contract
from multimethod import multimethod
from jaxtyping import ArrayLike
from jax.experimental import sparse
from jax.experimental.sparse._base import JAXSparse

MINVAL = jnp.finfo(float).eps

def stable_xlogx(x):
    # xlogxを計算する際に、最小値をクリップして安定性を確保
    return xlogy(x, jnp.clip(x, MINVAL))

def stable_entropy(x):
    # エントロピーを計算
    return - stable_xlogx(x).sum()

def stable_cross_entropy(x, y):
    # クロスエントロピーを計算
    return - xlogy(x, y).sum()

def log_stable(x):
    # 対数を計算する際に、最小値をクリップして安定性を確保
    return jnp.log(jnp.clip(x, min=MINVAL))


@multimethod
@partial(jit, static_argnames=["keep_dims"])
def factor_dot(M: ArrayLike, xs: list[ArrayLike], keep_dims: Optional[tuple[int]] = None):
    """Dot product of a multidimensional array with `x`.
    Parameters
    ----------
    - `qs` [list of 1D numpy.ndarray] - list of jnp.ndarrays

    Returns
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return factor_dot_flex(M, xs, dims, keep_dims=keep_dims)


@multimethod
def factor_dot(M: JAXSparse, xs: List[ArrayLike], keep_dims: Optional[Tuple[int]] = None):
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return spm_dot_sparse(M, xs, dims, keep_dims=keep_dims)


def spm_dot_sparse(
    X: JAXSparse, x: List[ArrayLike], dims: Optional[List[Tuple[int]]], keep_dims: Optional[List[Tuple[int]]]
):
    if dims is None:
        dims = (jnp.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    dims = jnp.array(dims).flatten()

    if keep_dims is not None:
        for d in keep_dims:
            if d in dims:
                dims = jnp.delete(dims, jnp.argwhere(dims == d))

    for d in range(len(x)):
        s = jnp.ones(jnp.ndim(X), dtype=int)
        s = s.at[dims[d]].set(jnp.shape(x[d])[0])
        X = X * x[d].reshape(tuple(s))

    sparse_sum = sparse.sparsify(jnp.sum)
    Y = sparse_sum(X, axis=tuple(dims))
    return Y


@partial(jit, static_argnames=["dims", "keep_dims"])
def factor_dot_flex(M, xs, dims: List[Tuple[int]], keep_dims: Optional[Tuple[int]] = None):
    """Dot product of a multidimensional array with `x`.

    Parameters
    ----------
    - `M` [numpy.ndarray] - tensor   b: p(s_{t+1}|s_t) --- 遷移行列 b
    - 'xs' [list of numpy.ndarray] - list of tensors xs: q(s_t) --- 各時刻の隠れ状態の分布 xs
    - 'dims' [list of tuples] - list of dimensions of xs tensors in tensor M --- 各因子ベクトルがMのどの軸に対応するかを示す dims
    - 'keep_dims' [tuple] - tuple of integers denoting dimensions to keep --- 出力で残す軸 keep_dims
    Returns
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    # B_marg.append( factor_dot_flex(b, xs, tuple(dims), keep_dims=keep_dims) )
    # b: 遷移行列のスタック
    # xs: f以外の親因子のqs
    # dims: f以外の親因子の軸（bにおける場所）を指定
    # keep_dims: 時間軸、子因子、親因子fの次元を残す

    all_dims = tuple(range(M.ndim)) # bの全ての軸を取得
    matrix = [[xs[f], dims[f]] for f in range(len(xs))] # 各親因子のqsと対応軸のペアを作成
    args = [M, all_dims]
    for row in matrix: #各親因子のqsと対応軸のペアを追加
        args.extend(row)

    args += [keep_dims] #最後に「出力で残す軸」を指定
    # 最終的に
    # args (b, bの全ての軸 (0,1,2,...),  親因子のqs qs1, 親因子が対応するbの軸 (3,), qs2, (4,), ..., 残す軸 (0,1,3))

    return contract(*args, backend="jax")
    # p(s_{t+1} \mid s_t^{(f)})= \sum_{s_t^{(i)}} p(s_{t+1} | s_t^{(f)}, s_t^{(i)}) q(s_t^{(i)})
    #   Y[t, s_{i,t+1}, s_{f,t}]
    #   = \sum_{d \in \text{parents} \setminus f}
    #   p(s_{i,t+1} | s_{f,t}, s_{d,t})
    #   \prod_{d \neq f} q_d(s_{d,t})
    #   = p(s_{i,t+1} | s_{f,t})

    # args = (b, (0,1,2,3),  q1, (2,),  q2, (3,), (0,1,3))
    # B の軸は (0,1,2,3) = (T, s_{t+1}, s_t^{(1)}, s_t^{(2)})
    # - q1 は軸 (2,) に対応（= 現在の因子1 の分布 q(s_t^{(1)})）
    # - q2 は軸 (3,) に対応（= 現在の因子2 の分布 q(s_t^{(2)})）
    # - 出力は (0,1,3) を 残す（= 時刻 T・次状態 s_{t+1}・現在の因子2 の状態 s_t^{(2)} は保持）
    #
    # Y = contract(
    #     b,  (0,1,2,3),
    #     q1, (2,),     
    #     (0,1,3)       
    # )
    #
    # B の形状は (T, Snext, S1, S2) で、各軸は次のように対応しています：
    # B_{t,s_{t+1},s_t^{(i)},s_t^{(j)}} = p(s_{t+1} | s_t^{(1)}, s_t^{(2)})
    #   軸番号
    #   t:0
    #   s_{t+1}:1
    #   s_t^{(1)}:2
    #   s_t^{(2)}:3
    # \sum_{s^{(1)}_t} p(s_{t+1} | s_t^{(1)}, s_t^{(2)}) q(s^{(1)}_t) = p(s_{t+1} | s_t^{(2)})


def get_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Return observation likelihood for a single observation modality m"""
    if distr_obs:
        # 確率分布が与えられた場合、観測o_mに対する尤度を計算。期待尤度
        # p(o_m|s) = Σ_o p(o_m) × p(o_m|s)
        expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
        likelihood = (expanded_obs * A_m).sum(axis=0)
    else:
        # 離散観測の場合
        # p(o_m|s) = A_m[o_m]
        likelihood = A_m[o_m]

    return likelihood

def compute_log_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Compute observation log-likelihood for a single modality"""
    return log_stable(get_likelihood_single_modality(o_m, A_m, distr_obs=distr_obs))


def compute_log_likelihood(obs, A, distr_obs=True):
    """Compute likelihood over hidden states across observations from different modalities"""
    result = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)
    ll = jnp.sum(jnp.stack(result), 0)

    return ll


def compute_log_likelihood_per_modality(obs, A, distr_obs=True):
    """Compute likelihood over hidden states across observations from different modalities, and return them per modality"""
    """異なるモダリティからの観測ごとに、隠れ状態に対する対数尤度を計算し、各モダリティごとに返す。"""
    
    ll_all = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs),
                                obs, # 各モダリティごとの観測（例: [obs1, obs2, ...]）
                                A) # 各モダリティごとの観測モデルA（例: [A1, A2, ...]）


    return ll_all #それぞれの $ll_m$ が**「隠れ状態 s の数だけの配列（またはテンソル）」


def compute_accuracy(qs, obs, A):
    """Compute the accuracy portion of the variational free energy (expected log likelihood under the variational posterior)"""

    log_likelihood = compute_log_likelihood(obs, A)

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = log_likelihood * x
    return joint.sum()


def compute_free_energy(qs, prior, obs, A):
    """
    Calculate variational free energy by breaking its computation down into three steps:
    1. computation of the negative entropy of the posterior -H[Q(s)]
    2. computation of the cross entropy of the posterior with the prior H_{Q(s)}[P(s)]
    3. computation of the accuracy E_{Q(s)}[lnP(o|s)]

    Then add them all together -- except subtract the accuracy
    """

    vfe = 0.0  # initialize variational free energy
    for q, p in zip(qs, prior):
        negH_qs = - stable_entropy(q)
        xH_qp = stable_cross_entropy(q, p)
        vfe += (negH_qs + xH_qp)
    
    vfe -= compute_accuracy(qs, obs, A)

    return vfe


def multidimensional_outer(arrs):
    """Compute the outer product of a list of arrays by iteratively expanding the first array and multiplying it with the next array"""
    # 外積を計算するために、最初の配列を繰り返し拡張し、次の配列と乗算する
    # 簡単なサンプルコードで外積していることを確認。

    x = arrs[0]
    for q in arrs[1:]:
        x = jnp.expand_dims(x, -1) * q

    return x


def _exact_wnorm(A):
    """
    Implements (-1) * eq. (D.15) in Da Costa et al. ‘Active inference on discrete state-spaces: A synthesis’, Journal of Mathematical Psychology, 2020.

    Note: Like the legacy SPM implementation this function clips A for numerical stability. However note that if some values of Aare set to zero e.g. by Bayesian model reduction, these are non-zeroed in this calculation, and thus contribute a large amount to the information gain unless these are zeroed when multiplying by beliefs about states and expected observations. In principle, this should be the case.
    """
    # Clip once and reuse for numerical stability
    safe_A = jnp.clip(A, MINVAL)
    safe_sumA = jnp.clip(safe_A.sum(axis=0), MINVAL)

    wA = (
        jnp.log(safe_sumA) - jnp.log(safe_A)
        + 1. / safe_A - 1. / safe_sumA
        + digamma(safe_A) - digamma(safe_sumA)
    )# digammaの計算はscipyに任せる

    return -wA # TODO: minus sign here gives negative info gain for backward compatibility with spm implementation. Later will need to remove minus sign here to get positive info gain and adjust function documentation accordingly.


def spm_wnorm(A):
    """
    Returns the weight matrix used in PyMDP's parameter information-gain term.

    Historically this was the heuristic ``1/Σα − 1/α``. If exact_param_info_gain is set to *True* we instead return the exact value of
    the weight matrix used in the info gain computation defined in _exact_wnorm 
    while keeping the original function signature so that the rest of the codebase remains unchanged.
    """
    # A: 観測モデルのパラメータ。規格化されていない。
    #
    # A.shape == (n_outcomes, S_dep1, S_dep2, ...)
    # # A の形: (outcomes, S1, S2)
    # A = np.array([
    #   # outcome 0
    #   [[0.2, 0.3],
    #    [0.1, 0.4]],
    #   # outcome 1
    #   [[0.8, 0.7],
    #    [0.9, 0.6]]
    # ])   # shape (2, 2, 2)
    # # A.sum(axis=0) は outcomes 軸を合計 -> shape (2,2)
    # A.sum(axis=0)
    # # => array([[1.0, 1.0],
    # #           [1.0, 1.0]])
    # # 各 (S1,S2) の組み合わせ（列に相当）について観測確率の総和を得る
    # print(A)

    if exact_param_info_gain:
        return _exact_wnorm(A)

    """spm legacy heuristic for computing information-gain over parameters:
    Implements (-2) * second line of eq. (D.17) in Da Costa et al. ‘Active inference on discrete state-spaces: A synthesis’, Journal of Mathematical Psychology, 2020"""
    norm = 1. / A.sum(axis=0)
    # 1/総和を計算する

    avg = 1. / (A + MINVAL)
    # 1/各値を計算する。MINVALを足してゼロ割りを防ぐ。

    wA = norm - avg
    # wA = 1/総和 - 1/各値

    # wAとは何だ？
    # 𝐾𝐿[𝐷𝑖𝑟(𝝁^((1) )∣𝜶′)||𝐷𝑖𝑟(𝝁^((1) )∣𝜶)] ]の粗い近似だと思う。
    # 𝐸_𝐷𝑖𝑟(𝜇^(s,1)∣𝛼^(𝑠,1) )  [ln⁡〖𝜇_𝑘^(𝑠,1) 〗 ]の𝜓(𝑥) の粗い近似とも一致する。
    # 前者ならNoveltyの計算と合うし、後者ならパラメタの期待値の計算と合う。
    # 𝜓(𝑎) \approx −𝛾−1/𝑥の近似を使うと対数項を無視することになり、
    # 𝜓(𝑎) \approx ln⁡𝑎−1/2𝑎の近似を使うと対数項がきれいに消えるが、wA = 1/2(norm - avg)になって合わない。
    # まいった。
    # ただ、1/2は定数だからいらないという見方もできる。

    return wA


def dirichlet_expected_value(dir_arr):
    """
    Returns Expectation of Dirichlet parameters over a set of
    Categorical distributions, stored in the columns of A.
    """
    dir_arr = jnp.clip(dir_arr, min=MINVAL) # ディリクレ分布のパラメータをクリップ
    expected_val = jnp.divide(dir_arr, dir_arr.sum(axis=0, keepdims=True)) # 期待値を計算
    return expected_val


if __name__ == "__main__":
    obs = [0, 1, 2]
    obs_vec = [nn.one_hot(o, 3) for o in obs]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs_vec, A)

    print(res)
