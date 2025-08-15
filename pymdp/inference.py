#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import jax.numpy as jnp
from pymdp.algos import run_factorized_fpi, run_mmp, run_vmp
from jax import tree_util as jtu, lax
from jax.experimental.sparse._base import JAXSparse
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike

eps = jnp.finfo('float').eps

def update_posterior_states(
    A,
    B,
    obs,
    past_actions,
    prior=None,
    qs_hist=None,
    A_dependencies=None,
    B_dependencies=None,
    num_iter=16,
    method="fpi",
):

    if method == "fpi" or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else: # VMP or MMP
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        if past_actions is not None:
            nf = len(B) #リストBの要素数、つまり因子の数
            actions_tree = [past_actions[:, i] for i in range(nf)]
            # Bの各因子に対して、過去の行動を抽出
            # 具体例（多対1）
            # セットアップ
            # 状態ファクター：
            #   s^0 = 位置（3状態: {A,B,C}）
            # 制御ファクター：
            #   c^0 = 水平: 2値 {Left=0, Right=1}
            #   c^1 = 垂直: 3値 {Stay=0, Up=1, Down=2}
            # 依存関係（多対1）：s^0 は c^0 と c^1 の両方に制御される
            # → B_action_dependencies[0] = [0, 1]
            # B の形（フラット化の前後）
            # フラット化前（概念的）：
            #   B[0].shape = (n_s', n_s_prev, n_u_horiz=2, n_u_vert=3)
            # フラット化後（実装の計算都合）：
            #   B[0].shape = (n_s', n_s_prev, n_u_flat=2*3=6)
            # ここで「行動軸」は6通りの複合行動（(水平,垂直) の全組み合わせ）に対応します。
            # インデックスの作り方は一貫した規約で固定します。
            # 例えば行動行列の次元がdims = [2, 3]（水平→垂直の順）なら、複合行動インデックスは
            #   u_flat = horiz * 3 + vert
            # となります（horiz ∈ {0,1}, vert ∈ {0,1,2} なので 0〜5 ）。
            # 各時刻 t の「生の行動」は、制御ファクターごとにこうだとします：
            #   t	水平 c^0	垂直 c^1
            #   0	Right=1	Down=2
            #   1	Left=0	Up=1
            #   2	Right=1	Stay=0
            #   3	Left=0	Stay=0
            # これを 状態ファクター f=0（=位置）に対する複合行動インデックスに変換します：
            # dims = [2, 3]  # 行動行列 [水平2値, 垂直3値] 
            # u_flat(t) = horiz(t) * 3 + vert(t)
            # 
            # t=0: 1*3 + 2 = 5
            # t=1: 0*3 + 1 = 1
            # t=2: 1*3 + 0 = 3
            # t=3: 0*3 + 0 = 0
            # 
            # past_actions は「因子 × 時間」で保持するので、ここに「位置」を動かす複合行動インデックスを入れます：
            # もし隠れ状態ファクターが nf=1（位置だけ）なら
            # past_actionsはT個の複合行動インデックスを持つ。
            # past_actions = [
            #     jnp.array([5, 1, 3, 0]),   # 因子 f=0（多対1依存でフラット化されたインデックス）
            #     jnp.array([0, 0, 1, 1])    # 因子 f=1（単一制御因子依存）
            # ]
            # past_actions[:, 0] = [5, 1, 3, 0]
            # もし他にも因子（例：バッテリー残量）があれば、その因子 f の B[f] の行動軸に合わせて（直積があれば同様にフラット化して）past_actions[:, f] に整数インデックスを入れます。
            # （制御されない因子は num_controls[f]=1 なので常に 0 で OK）
            #
            # つまり [past_actions[:, i] for i in range(nf)] は
            # 状態因子iについての過去の行動を抽出している。

            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(
                lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0),
                B,
                actions_tree,
            )
            # 元々のB[f].shape = (num_states[f], num_states[deps...], num_actions[f])
            # len(actions_tree) = nf、actions_tree[f].shape = (T,)：actions_treeは各時刻の行動に対応したインデックスが保存されているリスト。
            # 具体例
            # 簡単なB行列bと行動履歴a_idxで考えてみましょう。
            # b: 形状が(次状態:2, 現状態:2, 全行動:3)のB行列とします。3種類の行動（0, 1, 2）のルールが格納されています。
            # a_idx: [0, 2, 1, 0] はactions_treeから得られた、4ステップ分の行動履歴です。
            # このとき、b[..., a_idx] という処理は、
            # ...により、次元0（次状態）と次元1（現状態）はそのまま保持します。
            # a_idx [0, 2, 1, 0] を使って、最後の次元（全行動）から以下のようにスライスを抜き出します。
            #   0番目の時間ステップには、行動0のルール (b[:, :, 0]) を
            #   1番目の時間ステップには、行動2のルール (b[:, :, 2]) を
            #   2番目の時間ステップには、行動1のルール (b[:, :, 1]) を
            #   3番目の時間ステップには、行動0のルール (b[:, :, 0]) を
            # これらを一括で束ねて、新しいテンソルを作成します。
            # つまり、各時間ごとの遷移確率行列が得られる。
            # b[..., a_idx]という一回の操作で、JAX（NumPy）は各時間の遷移確率（行列）をすべて含んだ、新しい単一のテンソルを生成します。
            # b[..., a_idx].shape = (num_states[f], num_states[deps...], T)
            # jnp.moveaxisでTを前に持っていく。
            # よって、bの形は(T, num_states[f], num_states[deps...])となります。
            #
            # jtu.tree_mapの動作
            #   jtu.tree_mapは、入力されたリスト（Bとactions_tree）の要素をペアにして、関数（lambda式）を適用します。
            #   1回目のループ:
            #      b に B[0]（因子0のテンソル）が入る。
            #      a_idx に actions_tree[0]（因子0の行動履歴）が入る。
            #      lambda式がこれらを処理し、時間軸が先頭に来た新しい因子0のテンソルを生成する。
            #   2回目のループ:
            #   b に B[1]（因子1のテンソル）が入る。
            #       a_idx に actions_tree[1]（因子1の行動履歴）が入る。
            #       lambda式がこれらを処理し、時間軸が先頭に来た新しい因子1のテンソルを生成する。
            #   この処理が、全てのファクターについて繰り返されます。
            # 結果として、len(B) == nf となります。その中のテンソルb.shape(T, num_states[f], num_states[deps...])となる。

    
        else: # past_actions is None
            B = None

        # mmp, vmpのとき、Bには時間軸が追加されている。Bは遷移行列のスタックで、時間順に並んでいる。

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == "vmp":
            qs = run_vmp(
                A,
                B,
                obs,
                prior,
                A_dependencies,
                B_dependencies,
                num_iter=num_iter,
            )
        if method == "mmp":
            qs = run_mmp(
                A,
                B,
                obs,
                prior,
                A_dependencies,
                B_dependencies,
                num_iter=num_iter,
            )

    if qs_hist is not None:
        if method == "fpi" or method == "ovf":
            qs_hist = jtu.tree_map(
                lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0),
                qs_hist,
                qs,
            )
        else:
            # TODO: return entire history of beliefs
            qs_hist = qs
    else:
        if method == "fpi" or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            qs_hist = qs

    return qs_hist

def joint_dist_factor(b: ArrayLike, filtered_qs: list[Array], actions: Array):
    qs_last = filtered_qs[-1]
    qs_filter = filtered_qs[:-1]

    def step_fn(qs_smooth, xs):
        qs_f, action = xs
        time_b = b[..., action]
        qs_j = time_b * qs_f
        norm = qs_j.sum(-1, keepdims=True)
        if isinstance(norm, JAXSparse):
            norm = sparse.todense(norm)
        norm = jnp.where(norm == 0, eps, norm)
        qs_backward_cond = qs_j / norm
        qs_joint = qs_backward_cond * jnp.expand_dims(qs_smooth, -1)
        qs_smooth = qs_joint.sum(-2)
        if isinstance(qs_smooth, JAXSparse):
            qs_smooth = sparse.todense(qs_smooth)
        
        # returns q(s_t), (q(s_t), q(s_t, s_t+1))
        return qs_smooth, (qs_smooth, qs_joint)

    # seq_qs will contain a sequence of smoothed marginals and joints
    _, seq_qs = lax.scan(
        step_fn,
        qs_last,
        (qs_filter, actions),
        reverse=True,
        unroll=2
    )

    # we add the last filtered belief to smoothed beliefs

    qs_smooth_all = jnp.concatenate([seq_qs[0], jnp.expand_dims(qs_last, 0)], 0)
    qs_joint_all = seq_qs[1]
    if isinstance(qs_joint_all, JAXSparse):
        qs_joint_all.shape = (len(actions),) + qs_joint_all.shape
    return qs_smooth_all, qs_joint_all


def smoothing_ovf(filtered_post, B, past_actions):
    assert len(filtered_post) == len(B)
    nf = len(B)  # number of factors

    joint = lambda b, qs, f: joint_dist_factor(b, qs, past_actions[..., f])

    marginals_and_joints = ([], [])
    for b, qs, f in zip(B, filtered_post, list(range(nf))):
        marginals, joints = joint(b, qs, f)
        marginals_and_joints[0].append(marginals)
        marginals_and_joints[1].append(joints)

    return marginals_and_joints


    
