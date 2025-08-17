import jax.numpy as jnp
import jax.tree_util as jtu

from jax import jit, vmap, grad, lax, nn
# from jax.config import config
# config.update("jax_enable_x64", True)

from pymdp.maths import compute_log_likelihood, compute_log_likelihood_per_modality, log_stable, MINVAL, factor_dot, factor_dot_flex
from typing import Any, List

def add(x, y):
    return x + y

def marginal_log_likelihood(qs, log_likelihood, i):
    xs = [q for j, q in enumerate(qs) if j != i]
    return factor_dot(log_likelihood, xs, keep_dims=(i,))

def all_marginal_log_likelihood(qs, log_likelihoods, all_factor_lists):
    qL_marginals = jtu.tree_map(lambda ll_m, factor_list_m: mll_factors(qs, ll_m, factor_list_m), log_likelihoods, all_factor_lists)
    
    num_factors = len(qs)

    # insted of a double loop we could have a list defining m to f mapping
    # which could be resolved with a single tree_map cast
    qL_all = [jnp.zeros(1)] * num_factors
    for m, factor_list_m in enumerate(all_factor_lists):
        for l, f in enumerate(factor_list_m):
            qL_all[f] += qL_marginals[m][l]

    return qL_all

def mll_factors(qs, ll_m, factor_list_m) -> List:
    relevant_factors = [qs[f] for f in factor_list_m]
    marginal_ll_f = jtu.Partial(marginal_log_likelihood, relevant_factors, ll_m)
    loc_nf = len(factor_list_m)
    loc_factors = list(range(loc_nf))
    return jtu.tree_map(marginal_ll_f, loc_factors)

def run_vanilla_fpi(A, obs, prior, num_iter=1, distr_obs=True):
    """ Vanilla fixed point iteration (jaxified) """

    nf = len(prior)
    factors = list(range(nf))
    # Step 1: Compute log likelihoods for each factor
    ll = compute_log_likelihood(obs, A, distr_obs=distr_obs)
    # log_likelihoods = [ll] * nf

    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    # Step 3: Iterate until convergence
    def scan_fn(carry, t):
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        mll = jtu.Partial(marginal_log_likelihood, q, ll)
        marginal_ll = jtu.tree_map(mll, factors)
        log_q = jtu.tree_map(add, marginal_ll, log_prior)

        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    qs = jtu.tree_map(nn.softmax, res)
    return qs

def run_factorized_fpi(A, obs, prior, A_dependencies, num_iter=1):
    """
    Run the fixed point iteration algorithm with sparse dependencies between factors and outcomes (stored in `A_dependencies`)
    """

    # Step 1: Compute log likelihoods for each factor
    log_likelihoods = compute_log_likelihood_per_modality(obs, A)

    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    # Step 3: Iterate until convergence
    def scan_fn(carry, t):
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        marginal_ll = all_marginal_log_likelihood(q, log_likelihoods, A_dependencies)
        log_q = jtu.tree_map(add, marginal_ll, log_prior)

        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    qs = jtu.tree_map(nn.softmax, res)
    return qs

def mirror_gradient_descent_step(tau, ln_A, lnB_past, lnB_future, ln_qs):
    """
    u_{k+1} = u_{k} - \nabla_p F_k
    p_k = softmax(u_k)
    """
    err = ln_A - ln_qs + lnB_past + lnB_future
    # ln_qs:元の値
    # ln_A + lnB_past + lnB_future: メッセージから計算されたln_qs
    ln_qs = ln_qs + tau * err # 誤差を足す
    qs = nn.softmax(ln_qs - ln_qs.mean(axis=-1, keepdims=True)) #確率にする

    return qs

def update_marginals(get_messages, # メッセージ取得関数
                     obs,   # 観測値列
                     A,     # 観測モデル p(s|o)
                     B,     # 遷移モデル p(s_{t+1}|s_t, a_t)
                     prior, # 事前分布 p(s_0) D行列
                     A_dependencies, # 観測モデルの依存関係
                     B_dependencies, # 遷移モデルの依存関係
                     num_iter=1,     # イテレーション回数
                     tau=1.,):       # ステップサイズ、学習の進む速さ
    """" Version of marginal update that uses a sparse dependency matrix for A """

    T = obs[0].shape[0] # 観測系列の長さ（時系列長T）
    ln_B = jtu.tree_map(log_stable, B) 
    # 遷移行列Bを対数空間へ、単なるlogの変換ではなく、安定性のためにクリップを使用
    # ネストした構造（リスト、辞書、配列など）に対して指定した関数を再帰的に適用
    # 例えば、Bが複数の因子を持つ場合、各因子の遷移行列を対数空間に変換する
    # log likelihoods -> $\ln(A)$ for all time steps
    # for $k > t$ we have $\ln(A) = 0$

    def get_log_likelihood(obs_t, A):
       # # mapping over batch dimension
       # return vmap(compute_log_likelihood_per_modality)(obs_t, A)
       return compute_log_likelihood_per_modality(obs_t, A)

    # mapping over time dimension of obs array
    log_likelihoods = vmap(get_log_likelihood, (0, None))(obs, A) # this gives a sequence of log-likelihoods (one for each `t`)

    ln_qs = jtu.tree_map( lambda p: jnp.broadcast_to(jnp.zeros_like(p), (T,) + p.shape), prior)
    # ここでqsを初期化
    # log marginals -> $\ln(q(s_t))$ for all time steps and factors
    # 各因子の事前分布を対数空間に変換
    # 引数はqでpriorが渡される。
    # jnp.zeros_like(p) priorの形状に合わせて0で初期化
    #   例えば、priorが3つの因子を持つ場合、各因子の事前分布を0で初期化した配列を作成
    # jnp.broadcast_to(..., (T,) + p.shape)
    #   (T,) + p.shape とは「先頭に時系列長Tを追加したshape」という意味
    #   例：T=5、p.shape=(3,) なら (5, 3) になる
    # broadcast_to で「全ての時刻に同じゼロベクトル」をコピーする
    #   → 結果：5×3 の配列、全て0
    # jax.tree_map は、PyTree（複雑なリスト・タプル・辞書構造など） の中身（leaf）すべてに対して、同じ関数を適用して戻り値も同じ構造で返す関数
    # priorの中の各要素（leaf）に func(...) を適用し、その結果を同じ構造で返す
    #
    # 例
    # prior = [
    #     jnp.array([0.3, 0.5, 0.2]),  # 因子0: shape (3,)
    #     jnp.array([0.6, 0.4])        # 因子1: shape (2,)
    # ]
    # T=5なら
    # ln_qs = [
    #     jnp.zeros((5, 3)),  # 因子0: shape (T=5, S0=3)
    #     jnp.zeros((5, 2))   # 因子1: shape (T=5, S1=2)
    # ]
    # qs[f]なら因子fの時間ごとの信念分布が得られる。

    # log prior -> $\ln(p(s_t))$ for all factors
    ln_prior = jtu.tree_map(log_stable, prior)

    # qsをsoftmaxで正規化
    qs = jtu.tree_map(nn.softmax, ln_qs)


    def scan_fn(carry, iter):
        #qsを更新する関数
        qs = carry

        ln_qs = jtu.tree_map(log_stable, qs)
        # messages from future $m_+(s_t)$ and past $m_-(s_t)$ for all time steps and factors. For t = T we have that $m_+(s_T) = 0$
        
        lnB_past, lnB_future = get_messages(ln_B, B, qs, ln_prior, B_dependencies)

        mgds = jtu.Partial(mirror_gradient_descent_step, tau)
        # jtu.Partial
        # jax.tree_util.Partial は functools.partial に似ています。
        # 最初の引数（この場合は tau）を固定して、新しい関数を返します。
        # 返された関数は、残りの引数だけを渡せば呼び出せるようになります。

        ln_As = vmap(all_marginal_log_likelihood, in_axes=(0, 0, None))(qs, log_likelihoods, A_dependencies)
        # 観測からのメッセージ

        qs = jtu.tree_map(mgds, ln_As, lnB_past, lnB_future, ln_qs)
        # qsを更新する


        return qs, None

    # forループnum_iter回す
    qs, _ = lax.scan(scan_fn, qs, jnp.arange(num_iter))
    # scanを複数回呼び出しqsを更新する。
    # これにより、qsは各イテレーションで更新され、最終的な推論結果が得られる。 

    return qs

def variational_filtering_step(prior, Bs, ln_As, A_dependencies):

    ln_prior = jtu.tree_map(log_stable, prior)
    
    #TODO: put this inside scan
    ####
    marg_ln_As = all_marginal_log_likelihood(prior, ln_As, A_dependencies)

    # compute posterior q(z_t) -> n x 1 x d
    post = jtu.tree_map( 
            lambda x, y: nn.softmax(x + y, -1), marg_ln_As, ln_prior 
        )
    ####

    # compute prediction p(z_{t+1}) = \int p(z_{t+1}|z_t) q(z_t) -> n x d x 1
    pred = jtu.tree_map(
            lambda x, y: jnp.sum(x * jnp.expand_dims(y, -2), -1), Bs, post
        )
    
    # compute reverse conditional distribution q(z_t|z_{t+1})
    cond = jtu.tree_map(
        lambda x, y, z: x * jnp.expand_dims(y, -2) / jnp.expand_dims(z, -1),
        Bs,
        post, 
        pred
    )

    return post, pred, cond

def update_variational_filtering(obs, A, B, prior, A_dependencies, **kwargs):
    """Online variational filtering belief update that uses a sparse dependency matrix for A"""

    T = obs[0].shape[0]
    def pad(x):
        npad = [(0, 0)] * jnp.ndim(x)
        npad[0] = (0, 1)
        return jnp.pad(x, npad, constant_values=1.)
    
    B = jtu.tree_map(pad, B)
 
    def get_log_likelihood(obs_t, A):
        # mapping over batch dimension
        return vmap(compute_log_likelihood_per_modality)(obs_t, A)

    # mapping over time dimension of obs array
    log_likelihoods = vmap(get_log_likelihood, (0, None))(obs, A) # this gives a sequence of log-likelihoods (one for each `t`)
    
    def scan_fn(carry, iter):
        _, prior = carry
        Bs, ln_As = iter

        post, pred, cond = variational_filtering_step(prior, Bs, ln_As, A_dependencies)
        
        return (post, pred), cond

    init = (prior, prior)
    iterator = (B, log_likelihoods)
    # get q_T(s_t), p_T(s_{t+1}) and the history q_{T}(s_{t}|s_{t+1})q_{T-1}(s_{t-1}|s_{t}) ...
    (qs, ps), qss = lax.scan(scan_fn, init, iterator)

    return qs, ps, qss

def get_vmp_messages(ln_B, B, qs, ln_prior, B_dependencies):
    
    num_factors = len(qs)
    factors = list(range(num_factors))
    get_deps = lambda x, f_idx: [x[f] for f in f_idx] # function that effectively "slices" a list with a set of indices `f_idx`

    # make a list of lists, where each list contains all dependencies of a factor except itself
    all_deps_except_f = jtu.tree_map( 
        lambda f: [d for d in B_dependencies[f] if d != f], 
        factors
    )

    # make list of integers, where each integer is the position of the self-factor in its dependencies list
    position = jtu.tree_map(
        lambda f: B_dependencies[f].index(f),
        factors
    )

    if ln_B is not None:
        ln_B_marg = jtu.tree_map( # this is a list of matrices, where each matrix is the marginal transition tensor for factor f
            lambda b, f: factor_dot(b, get_deps(qs, all_deps_except_f[f]), keep_dims=(0, 1, 2 + position[f])), 
            ln_B, 
            factors
        )  # shape = (T, states_f_{t+1}, states_f_{t})
    else:
        ln_B_marg = None

    def forward(ln_b, q, ln_prior):
        msg = vmap(lambda x, y: y @ x)(q[:-1], ln_b) # ln_b has shape (num_states, num_states) qs[:-1] has shape (T-1, num_states)
        return jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0)
    
    def backward(ln_b, q):
        # q_i B_ij
        msg = vmap(lambda x, y: x @ y)(q[1:], ln_b)
        return jnp.pad(msg, ((0, 1), (0, 0)))

    if ln_B_marg is not None:
        lnB_future = jtu.tree_map(forward, ln_B_marg, qs, ln_prior)
        lnB_past = jtu.tree_map(backward, ln_B_marg, qs)
    else:
        lnB_future = jtu.tree_map(lambda x: 0., qs)
        lnB_past = jtu.tree_map(lambda x: 0., qs)
    
    return lnB_future, lnB_past 

def run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=1, tau=1.):
    '''
    Run variational message passing (VMP) on a sequence of observations
    '''

    qs = update_marginals(
        get_vmp_messages, 
        obs, 
        A, 
        B, 
        prior, 
        A_dependencies, 
        B_dependencies, 
        num_iter=num_iter, 
        tau=tau
    )
    return qs

def get_mmp_messages(ln_B,
                     B, # 遷移行列p(s_{t+1}|s_t, a_t)
                     qs,#q(s_t) --- 各時刻の隠れ状態の分布1<=t<=T
                     ln_prior, # ln p(s_0) --- 初期状態の事前分布
                     B_deps):  # 遷移モデルの依存関係。因子sごとに隠れマルコフモデルを想定して、メッセージパッシングを行うが、依存性のある因子同士は接続しており、メッセージの伝播が起こる。時刻t+1の各因子（子ファクター）に対し、それが依存する時刻tの因子（親ファクター）のインデックスをリストである。
    """Get messages for marginal message passing (MMP)"""
    """ 各関数で共有して使われているが引数で渡されている。"""

    num_factors = len(qs) # 各因子の数
    factors = list(range(num_factors)) # 各因子のインデックス

    get_deps_forw = lambda x, f_idx: [x[f][:-1] for f in f_idx]
    # 前方のqsを取得する関数
    # x, f_idxが引数
    # x: 各因子の信念分布qs
    # f_idx: 依存されている時刻tの因子（親ファクター）インデックス
    # x[f][:-1]
    # x[f]: 依存されている因子(親ファクター)fの信念ベクトル（時系列データ）を取り出します。
    # [:-1]: Pythonのスライス記法で、「最初から、最後の一つ手前まで」を意味します。これにより、信念ベクトルの最後の時間ステップが切り捨てられます。
    # qs = [
    #     jnp.zeros((5, 3)),  # 因子0: shape (T=5, S0=3)
    #     jnp.zeros((5, 2))   # 因子1: shape (T=5, S1=2)
    # ]

    get_deps_back = lambda x, f_idx: [x[f][1:] for f in f_idx]
    # 後方のqsを取得する関数
    # x, f_idxが引数
    # x: 各因子の信念分布qs
    # f_idx: 依存されている時刻tの因子（親ファクター）インデックス
    # x[f][1:]
    # x[f]: 依存されている因子(親ファクター)fの信念ベクトル（時系列データ）を取り出します。
    # [1:]: Pythonのスライス記法で、「1から最後まで」を意味します。これにより、信念ベクトルの最初の時間ステップが切り捨てられます。

    def forward(b, ln_prior, f): # 前方メッセージを計算
        xs = get_deps_forw(qs, B_deps[f])
        # qsを取得する。取得するモダリティは B_deps[f] で指定される。
        # 例えば、B_deps[f] が [0, 1] の場合
        # qs[0][:-1] と qs[1][:-1] を取得する。

        dims = tuple((0, 2 + i) for i in range(len(B_deps[f])))
        # dimsは、各因子の次元を指定するタプル。len(B_deps[f]) で依存する因子の数がわかり、それが次元数である。。

        msg = log_stable(factor_dot_flex(b, xs, dims, keep_dims=(0, 1) ))
        # b: p(s_{t+1}|s_t, a_t) の遷移行列, xs: q(s_t)因子の信念
        # dims: 「`xs` のどのベクトルを、`b` のどの軸に当てて縮約するか」の対応（`(0, 2+i)` のペアは“時間軸0と、bの(2+i)番目の因子軸を対応づけて足し込む”という指定）
        # `keep_dims=(0, 1)`: **時間軸(0) と “次状態”軸(1) だけ残し、他の軸（現状態たち）は総和で消す**指定
        # msg[t, s_{t+1}] = 
        # \sum_{依存する各因子のs_t} b[t, s_{t+1}, s_t^{(1)}, s_t^{(2)},\ldots];
        # \prod_k q^{(k)}[t,\, s_t^{(k)}]
        # 周辺化
        # append log_prior as a first message 

        msg = jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0)
        # jnp.expand_dims(ln_prior, 0)は (1, D) の 2D 配列1 行 D 列の行ベクトル
        # msgは (T, D) の 2D 配列で、Tは時系列長、Dは因子の次元数
        #jnp.concatenate([...], axis=0)
        #axis=0 は「行方向（縦方向）」に配列を繋ぐことを意味します。
        #[A, B] のようにリストで渡すと、それらの配列を縦に継ぎ足すように結合します。どちらも同じ列数（同じ第二軸のサイズ）でなければなりません。
        # 元の ln_prior: [a, b, c]（形 (3,)）
        # msg: 2 × 3 行列
        # これを実行すると：
        # [[a, b, c],
        #  [...元の msg の 1 行目...],
        #  [...元の msg の 2 行目...]]
        # ln_priorをメッセージの先頭に入れた。

        # mutliply with 1/2 all but the last msg
        T = len(msg)
        if T > 1:
            msg = msg * jnp.pad( 0.5 * jnp.ones(T - 1), (0, 1), constant_values=1.)[:, None]
        # 0.5 * jnp.ones(T - 1) は (T - 1,) の配列で、各要素が 0.5になる。
        # jnp.pad(..., (0, 1), constant_values=1.) は、paddingを行う。
        # 配列の最後に 1 を追加して (T,) の形になる。
        # 最初の T-1 行は 0.5 倍され、
        # 最後の行だけはそのまま（1.0 倍）になる。終端だからか？
        # これにより、msg の形は (T, D) から (T, D) のままとなります。

        return msg

    def backward(Bs, xs): # 後方メッセージを計算
        # Bs: 遷移行列のスタックではあるが、B_margである。
        # 例えば、B_margは以下のようなリストになる。
        # 親因子f、子因子iが1と2に依存している場合、
        # [
        #   jnp.zeros((5, 3, 2)),  # 因子0: shape (T=5, S0=3, Sf=2)
        #   jnp.zeros((5, 2, 2))   # 因子1: shape (T=5, S1=2, Sf=2)
        # ]
        # xs: 子因子の信念分布(t=1:T)。0は除外されている。
        msg = 0.
        for i, b in enumerate(Bs):
            #bは子因子iの時系列で並んだ遷移行列スタック
            #margeされている。
            
            b_norm = b / (b.sum(-1, keepdims=True) + 1e-16)
            # 規格化
            # Bは規格化された遷移行列だが、後ろ向き計算に使うBは転置して、列について規格化したものだから再規格化が必要になる。

            msg += log_stable(vmap(lambda x, y: y @ x)(b_norm, xs[i])) * .5
            # 1/2が掛けられている。なぜか0.5の記述がforwardと異なる。同じにしたほうがよいのでは。
            # xに遷移行列の規格化された転置行列が入る。
            # yに因子の信念分布が入る。
            # vmapで各時間の計算が並列で回る。
            # y @ x -> B^\dagger qs
            # 内積の順番を入れ替えることでB^\dagger qsを実現している。
        
        # ループで子因子iの遷移行列を順に取り出し、規格化してから、因子の信念分布と掛け算し、めっせーじを計算している。
        # それらの総和をとり、各時間のメッセージを計算している。
            
        
        return jnp.pad(msg, ((0, 1), (0, 0)))

    def marg(inv_deps, f):
        # inv_deps: 逆依存関係リスト。時刻tの因子（親ファクター）に依存されている時刻t+1の隠れ状態(小ファクター)のリスト。
        # f: 親因子インデックス（整数）
        B_marg = []
        for i in inv_deps:
            # i: 依存する因子（子ファクター）t+1のインデックス
            b = B[i] # 因子iに関する時間順にスタックされた遷移行列
            keep_dims = (0, 1, 2 + B_deps[i].index(f))
            # B_deps[i].index(f) B_deps[i]の中にある親ファクターfの位置（インデックス）
            # どの次元を残すかを指定
            # bは時間順にならんだ遷移行列のスタックになっている。
            # b[i] の軸の並びをイメージ：
            # 0: T 時間
            # 1: s_{t+1}^{(f)}
            # 2: s_{t}^{(i)}
            # 3: s_{t}^{(i+3)}
            # 2 + B_deps[i].index(f): 依存されている各因子の「現在状態」軸（B_deps[i].index(f) は因子fのB_deps[i] 内の位置）
            # 子因子1が1と3に親因子に依存する場合、B_deps[1]は[1, 3]となる。
            # 3の子因子を残すなら、2+3で5が残る。つまり、keep_dims=(0,1,5)になる。

            dims = []
            idxs = []
            for j, d in enumerate(B_deps[i]):
                # iが依存する親因子のリストをfor文で回す
                # j: リスト内のインデックス
                # d: 時刻tの親因子（いわゆる親ファクター）
                if f != d:
                    dims.append((0, 2 + j)) # 時間軸、親因子軸
                    idxs.append(d)          # 依存されている親因子（親ファクター）
            xs = get_deps_forw(qs, idxs)
            # qs: q(s_t^1), q(s_t^2),...
            # idxs: 時刻tの親因子のインデックス
            # xs: 依存される親因子の信念分布ではあるが、最後の時間がない。
            # qs = [
            #     jnp.zeros((5, 3)),  # 因子0: shape (T=5, S0=3)
            #     jnp.zeros((5, 2))   # 因子1: shape (T=5, S1=2)
            # ]
            # idxs=[2]なら
            # xs = p[jnp.zeros((4, 2))]

            B_marg.append( factor_dot_flex(b, xs, tuple(dims), keep_dims=keep_dims) )
            # b: 遷移行列のスタック
            # xs: 親因子のqs
            # dims: 親因子の軸（bにおける場所）を指定
            # keep_dims: 時間軸、親因子、子因子fの次元を残す
            # B_margは、周辺化した遷移行列のリストp(s_{i,t+1} | s_{f,t})
            # 因子は複数の因子に依存している。他の因子を信念を掛け同時確率にし、周辺化して消す。
            # p(s_{i,t+1} | s_{f,t}, s_{j,t})q(s_{j,t}) -> p(s_{i,t+1}, s_{j,t} | s_{f,t}) -> p(s_{i,t+1} | s_{f,t})

        # 例えば、B_margは以下のようなリストになる。
        # 親因子f、子因子iが1と2に依存している場合、
        # [
        #   jnp.zeros((5, 3, 2)),  # 因子0: shape (T=5, S0=3, Sf=2)
        #   jnp.zeros((5, 2, 2))   # 因子1: shape (T=5, S1=2, Sf=2)
        # ]

        return B_marg

    if B is not None:
        inv_B_deps = [[i for i, d in enumerate(B_deps) if f in d] for f in factors]
        # B_deps = [
        #   [0],       # 因子0は自分自身にだけ依存
        #   [0, 1],    # 因子1は因子0と1に依存
        #   [1, 2]     # 因子2は1と自分自身に依存
        # ]
        # enumerateでインデックスと要素を取得
        # factors 各因子のインデックス
        # f: 因子のインデックス
        # i: 依存関係のインデックス s_t^{(i)}
        # d: iが依存する因子のインデックス s_t^{(k)}
        # fがdに含まれるならばiを保存
        # 例えば、f=0のとき、
        #ループ1回目: dは[0]。 0 in [0] は True。 → iである0を保存。
        #ループ2回目: dは[0, 1]。 0 in [0, 1] は True。 → iである1を保存。
        #ループ3回目: dは[1, 2]。 0 in [1, 2] は False。
        # [0, 1]がリストに追加される。
        # これにより、時刻tの各因子（親ノード）に依存する時刻t+1の因子（子ノード）のインデックスを取得できる。
        # -------
        # forループで元のリストを回す
        # for <要素> in <元のリスト>:
        #   if文で条件をチェック
        #      if <条件式>:        
        #           条件に合ったら、新しいリストに追加
        #           new_list.append(<出力したい式>)
        # new_list = [ <出力したい式> for <要素> in <元のリスト> if <条件式> ]
        #
        # 逆依存関係がなぜ必要か？
        # 後方メッセージを計算するために、時刻tの因子が,時刻t+1のどの因子に依存しているかを知る必要がある。


        B_marg = jtu.tree_map(lambda f: marg(inv_B_deps[f], f), factors)
        # B_marg = []
        # for f in factors:
        #     result = marg(inv_B_deps[f], f)
        #     B_marg.append(result)
        # p(s_{i,t+1} | s_{f,t})

        lnB_future = jtu.tree_map(forward, B, ln_prior, factors) #forward messages
        # B: 観測モデルの遷移行列、各因子ごとの時間順にスタックされた遷移行列
        # ln_prior: 事前分布の対数確率
        # factors: 各因子のインデックス

        lnB_past = jtu.tree_map(lambda f: backward(B_marg[f], get_deps_back(qs, inv_B_deps[f])), factors) #backward messages
        # B_marg[f]: 周辺化された遷移行列
        # get_deps_back(qs, inv_B_deps[f]): 子因子（未来）の信念分布を取得
        # 各因子について、後方メッセージを計算する。


    else: 
        lnB_future = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), ln_prior)
        lnB_past = jtu.tree_map(lambda x: 0., qs)

    return lnB_future, lnB_past

def run_mmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=1, tau=1.):
    "周辺化メッセージパッシング (MMP) の実行"
    qs = update_marginals(
        get_mmp_messages, # メッセージ取得関数を渡す
        obs, # 観測値
        A,   # p(s|o) --- 観測モデル
        B,   # p(s_{t+1}|s_t) --- 遷移モデルではあるが、もとのBことなり、過去の行動に基づき並べられた遷移行列のスタックになっている。
        prior, #p(s_0) --- 事前分布
        A_dependencies, # 観測モデルの依存関係。Aは全因子sの次元を持つが、モダリティはすべての因子に依存するわけではない。モダリティと因子の依存性をA_dependenciesで指定する。p(o|s1, s2, ..., sn)と書けるが、実際にはp(o|s1, s2, ..., sn)のうち、s1, s2, ..., snのうち一部の因子に依存する。そこで、A_dependenciesを使って、どの因子がどのモダリティに依存するかを指定する。
        B_dependencies, # 遷移モデルの依存関係。Bは因子sごとに用意する。sそれぞれが、異なる因子に依存する。すなわち、Bはsごとに異なる大きさの遷移行列を持つ。B_dependenciesは、どの因子がどの因子に依存するかを指定し、これを見ることで、B行列がどの因子に依存するかを知ることができる。
        num_iter=num_iter, # イテレーション回数
        tau=tau # ステップサイズ、学習の進む速さ
    )
    return qs

def run_online_filtering(A, B, obs, prior, A_dependencies, num_iter=1, tau=1.):
    """Runs online filtering (HAVE TO REPLACE WITH OVF CODE)"""
    qs = update_marginals(get_mmp_messages, obs, A, B, prior, A_dependencies, num_iter=num_iter, tau=tau)
    return qs 

if __name__ == "__main__":
    prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(jnp.array([0, -80., -80., -80, -80.]))]
    obs = [nn.one_hot(0, 5), nn.one_hot(5, 10)]
    A = [jnp.ones((5, 2, 2, 5))/5, jnp.ones((10, 2, 2, 5))/10]
    
    qs = jit(run_vanilla_fpi)(A, obs, prior)

    # test if differentiable
    from functools import partial

    def sum_prod(prior):
        qs = jnp.concatenate(run_vanilla_fpi(A, obs, prior))
        return (qs * log_stable(qs)).sum()

    print(jit(grad(sum_prod))(prior))

    # def sum_prod(precision):
    #     # prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(log_prior)]
    #     prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(precision*nn.one_hot(0, 5))]
    #     qs = jnp.concatenate(run_vanilla_fpi(A, obs, prior))
    #     return (qs * log_stable(qs)).sum()

    # precis_to_test = 1.
    # print(jit(grad(sum_prod))(precis_to_test))

    # log_prior = jnp.array([0, -80., -80., -80, -80.])
    # print(jit(grad(sum_prod))(log_prior))

