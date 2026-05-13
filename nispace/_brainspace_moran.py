"""
Implementation of Moran spectral randomization.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# Copied from brainspace.null_models.moran.MoranRandomization, commit e64d065
# Full License:
# BSD 3-Clause License

# Copyright (c) 2019, The BrainSpace developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.sparse as ssp
import scipy.linalg
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.spatial.distance import cdist

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator


def compute_mem(w, spectrum='nonzero', tol=1e-6, n_components=None):
    """ Compute Moran eigenvectors map.

    Parameters
    ----------
    w : ndarray or sparse matrix, shape = (n_vertices, n_vertices)
        Spatial weight matrix. Accepts dense ndarray or any scipy sparse
        format. Sparse input triggers memory-efficient truncated
        eigendecomposition (requires ``n_components``).
    spectrum : {'all', 'nonzero'}, optional
        Eigenvalues/vectors to select. If 'all', recover all eigenvectors
        except the smallest one. Otherwise, select all except non-zero
        eigenvectors. Default is 'nonzero'. Ignored when ``n_components``
        is set (truncated path always returns nonzero eigenvectors).
    tol : float, optional
        Minimum value for an eigenvalue to be considered non-zero.
        Default is 1e-6.
    n_components : int or None, optional
        If set, compute only the top ``n_components`` eigenvectors via
        truncated eigendecomposition. Required for sparse input.
        Default is None (full decomposition, dense only).

    Returns
    -------
    mem : 2D ndarray, shape (n_vertices, n_components)
        Eigenvectors of the weight matrix in descending eigenvalue order.
    ev : 1D ndarray, shape (n_components,)
        Eigenvalues in descending order.

    References
    ----------
    * Wagner H.H. and Dray S. (2015). Generating spatially constrained
      null models for irregularly spaced data using Moran spectral
      randomization methods. Methods in Ecology and Evolution, 6(10):1169-78.
    """

    if spectrum not in ['all', 'nonzero']:
        raise ValueError("Unknown spectrum '{0}'.".format(spectrum))

    n = w.shape[0]

    if ssp.issparse(w):
        # Sparse path: use a LinearOperator for doubly-centered matvec so
        # we never materialize the dense Wc (n x n) matrix.
        # Wc @ v = W@v - sum(v)*m - dot(m,v)*ones + grand_mean*sum(v)*ones
        # where m = column-mean vector (1D, length n).
        if n_components is None:
            raise ValueError("'n_components' must be set for sparse weight matrices.")
        w = w.astype(np.float64)
        m = np.asarray(w.mean(axis=0)).ravel()          # (n,)
        grand_mean = m.mean()
        ones = np.ones(n, dtype=np.float64)

        def _matvec(v):
            v = np.asarray(v, dtype=np.float64)
            sv = v.sum()
            return (w @ v) - sv * m - np.dot(m, v) * ones + grand_mean * sv * ones

        op = LinearOperator((n, n), matvec=_matvec, dtype=np.float64)
        ev, mem = eigsh(op, k=n_components, which='LM')
        # sort descending
        order = np.argsort(ev)[::-1]
        ev, mem = ev[order], mem[:, order]
        # drop near-zero eigenvalues
        mask_nonzero = np.abs(ev) >= tol
        ev, mem = ev[mask_nonzero], mem[:, mask_nonzero]
        return mem.astype(np.float32), ev.astype(np.float32)

    # Dense path
    m = w.mean(axis=0, keepdims=True)
    wc = w.mean() - m - m.T
    wc += w

    if n_components is not None:
        # Truncated dense: compute only top n_components eigenvectors
        wc32 = wc.astype(np.float32)
        n_comp = min(n_components, n - 1)
        ev, mem = scipy.linalg.eigh(
            wc32, subset_by_index=[n - n_comp, n - 1]
        )
        ev, mem = ev[::-1], mem[:, ::-1]
        mask_nonzero = np.abs(ev) >= tol
        ev, mem = ev[mask_nonzero], mem[:, mask_nonzero]
        return mem, ev

    # Full dense decomposition (original path)
    ev, mem = np.linalg.eigh(wc.astype(np.float32))
    ev, mem = ev[::-1], mem[:, ::-1]

    ev_abs = np.abs(ev)
    mask_zero = ev_abs < tol
    n_zero = np.count_nonzero(mask_zero)

    if n_zero == 0:
        raise ValueError('Weight matrix has no zero eigenvalue.')

    if spectrum == 'all':
        if n_zero > 1:
            memz = np.hstack([mem[:, mask_zero], np.ones((n, 1))])
            q, _ = np.linalg.qr(memz)
            mem[:, mask_zero] = q[:, :-1]
            idx_zero = mask_zero.argmax()
        else:
            idx_zero = ev_abs.argmin()

        ev[idx_zero:-1] = ev[idx_zero + 1:]
        mem[:, idx_zero:-1] = mem[:, idx_zero + 1:]
        ev = ev[:-1]
        mem = mem[:, :-1]

    else:  # nonzero only
        mask_nonzero = ~mask_zero
        ev = ev[mask_nonzero]
        mem = mem[:, mask_nonzero]

    return mem, ev


def _rand_orthogonal(m, rng):
    """Haar-random m x m orthogonal matrix."""
    H = rng.standard_normal((m, m))
    Q, _ = np.linalg.qr(H) # QR -> orthonormal columns
    # make determinant +1  (optional)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def moran_randomization(x, mem, mev,
                        n_nulls=1000,
                        procedure='singleton',   # + 'rotate'
                        joint=False,
                        tol_block=1e-3,
                        seed=None):
    """ Generate random samples from `x` based on Moran spectral randomization.

    Parameters
    ----------
    x : 1D or 2D ndarray, shape = (n_vertices,) or (n_vertices, n_feat)
        Array of variables arranged in columns, where `n_feat` is the number
        of variables.
    mem : 2D ndarray, shape = (n_vertices, nv)
        Moran eigenvectors map, where `nv` is the number of eigenvectors
        arranged in columns.
    n_nulls : int, optional
        Number of random samples. Default is 1000.
    procedure : {'singleton, 'pair', 'rotate'}, optional
        Procedure to generate the random samples. Default is 'singleton'.
    joint : boolean, optional
        If True variables are randomized jointly. Otherwise, each variable is
        randomized separately. Default is False.
    tol_block : float, optional
        Minimum value for an eigenvalue to be considered non-zero.
        Default is 1e-3.
    seed : int or None, optional
        Random state. Default is None.
    
    Returns
    -------
    output : ndarray, shape = (n_rep, n_vertices, n_feat)
        Random samples. If ``n_feat == 1``, shape = (n_rep, n_vertices).

    See Also
    --------
    :func:`.compute_mem`
    :class:`.MoranRandomization`

    References
    ----------
    * Wagner H.H. and Dray S. (2015). Generating spatially constrained
      null models for irregularly spaced data using Moran spectral
      randomization methods. Methods in Ecology and Evolution, 6(10):1169-78.

    """
    
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None] # (N, 1)

    procedure = procedure.lower()
    if procedure not in ['singleton', 'pair', 'rotate']:
        raise ValueError(f"Unknown procedure '{procedure}'")

    rng = np.random.default_rng(seed)
    n_v, n_f = x.shape
    n_comp = mem.shape[1]
    n_cols = 1 if joint else n_f

    # ---- coefficient representation --------------------------------------------------------------
    coeff = mem.T @ (x - x.mean(0)) / x.std(0, ddof=1) # (n_comp, n_f)

    out = np.empty((n_nulls, n_v, n_f), dtype=np.float32)

    # ---- pre-compute degeneration blocks ---------------------------------------------------------
    blocks, start = [], 0
    for i in range(1, n_comp):
        if abs(mev[i] - mev[i-1]) > tol_block:
            blocks.append(np.arange(start, i))
            start = i
    blocks.append(np.arange(start, n_comp))

    # ---- null loop -------------------------------------------------------------------------------
    for r in range(n_nulls):
        C = coeff.copy()

        # ---- random ±1 with optional broadcasting ------------------------------------------------
        if procedure in ('singleton', 'pair'):
            signs = rng.choice([-1., 1.], size=(n_comp, n_cols))
            if joint:
                signs = np.broadcast_to(signs, (n_comp, n_f))
            C *= signs

            # ---- optional 'pair' mixing ----------------------------------------------------------
            if procedure == 'pair':
                pairs  = rng.permutation(n_comp)[: (n_comp // 2) * 2].reshape(-1, 2)
                phi    = rng.uniform(0, 2 * np.pi, size=(pairs.shape[0], n_cols))
                if joint:
                    phi = phi + np.arctan2(C[pairs[:, 0]], C[pairs[:, 1]])
                    phi = np.broadcast_to(phi, (pairs.shape[0], n_f))

                for (a, b), ang in zip(pairs, phi):
                    A, B = C[[a, b]]
                    C[a] =  np.cos(ang) * A + np.sin(ang) * B
                    C[b] = -np.sin(ang) * A + np.cos(ang) * B

        # ---- 'rotate' ----------------------------------------------------------------------------
        else:  
            #n_blk, n_flip = 0, 0
            for blk in blocks:
                m = len(blk)
                if m == 1: # singleton -> fallback to sign flip
                    s = rng.choice([-1, 1], size=(1, n_cols))
                    if joint:
                        s = np.broadcast_to(s, (1, n_f))
                    C[blk] *= s
                    #n_flip += 1
                else: # rotate block
                    R = _rand_orthogonal(m, rng) # (m, m)
                    C[blk] = R @ C[blk]   
                    #n_blk += 1
            #print(f"rotations: {n_blk}/{len(blocks)}, sign flips: {n_flip}/{len(blocks)}")
            
        # ---- back-projection ---------------------------------------------------------------------
        sim = mem @ C * x.std(0, ddof=1) + x.mean(0)
        out[r] = sim

    return out.squeeze() # (n_rep, n_v) or (n_rep, n_v, n_feat)



class MoranRandomization(BaseEstimator):
    """ Moran spectral randomization.

    Parameters
    ----------
    procedure : {'singleton, 'pair'}, optional
        Procedure to generate the random samples. Default is 'singleton'.
    spectrum : {'all', 'nonzero'}, optional
        Eigenvalues/vectors to select. If 'all', recover all eigenvectors
        except one. Otherwise, select all except non-zero eigenvectors.
        Default is 'nonzero'.
    joint : boolean, optional
        If True variables are randomized jointly. Otherwise, each variable is
        randomized separately. Default is False.
    n_nulls : int, optional
        Number of randomizations. Default is 1000.
    tol : float, optional
        Minimum value for an eigenvalue to be considered non-zero.
        Default is 1e-6.
    tol_block : float, optional
        Minimum value for an eigenvalue to be considered non-zero.
        Default is 1e-3.
    seed : int or None, optional
        Random state. Default is None.

    Attributes
    ----------
    mev_ : 1D ndarray, shape (n_components,)
        Eigenvalues of the weight matrix in descending order.
    mem_ : 2D ndarray, shape (n_vertices, n_components)
        Eigenvectors of the weight matrix in same order.

    See Also
    --------
    :class:`.SpinPermutations`

    """

    def __init__(self, procedure='singleton', spectrum='nonzero', joint=False,
                 n_nulls=1000, tol=1e-6, tol_block=1e-3, n_components=None,
                 seed=None):

        self.procedure = procedure
        self.spectrum = spectrum
        self.joint = joint
        self.n_nulls = n_nulls
        self.tol = tol
        self.tol_block = tol_block
        self.n_components = n_components
        self.seed = seed


    def fit(self, w):
        """ Compute Moran eigenvectors map.

        Parameters
        ----------
        w : BSPolyData, ndarray or sparse matrix, shape = (n_verts, n_verts)
            Spatial weight matrix or surface. If surface, the weight matrix is
            built based on the inverse geodesic distance between each vertex
            and the vertices in its `n_ring`.

        Returns
        -------
        self : object
            Returns self.

        """

        self.mem_, self.mev_ = compute_mem(w, spectrum=self.spectrum,
                                           tol=self.tol,
                                           n_components=self.n_components)
        return self


    def randomize(self, x):
        """ Generate random samples from `x`.

        Parameters
        ----------
        x : 1D or 2D ndarray, shape = (n_verts,) or (n_verts, n_feat)
            Array of variables arranged in columns, where `n_feat` is the
            number of variables.

        Returns
        -------
        output : ndarray, shape = (n_rep, n_verts, n_feat)
            Random samples. If ``n_feat == 1``, shape = (n_rep, n_verts).

        """

        rand = moran_randomization(x, self.mem_, self.mev_, n_nulls=self.n_nulls,
                                   procedure=self.procedure, joint=self.joint,
                                   tol_block=self.tol_block,
                                   seed=self.seed)
        return rand