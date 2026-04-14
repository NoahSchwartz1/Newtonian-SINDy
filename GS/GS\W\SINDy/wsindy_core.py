"""
wsindy_core.py
──────────────
Python implementation of WSINDy for ODEs.
Ported from MathBioCU/PyWSINDy_ODE (Messenger & Bortz, 2021).
"""

import numpy as np
import itertools
import operator
from scipy.linalg import lstsq
from numpy import matlib as mb
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class WSINDy:
    """Weak SINDy for ODEs.

    Parameters
    ----------
    polys : array-like
        Monomial powers to include (e.g. [0,1,2,3]).
    trigs : array-like
        Sine/cosine frequencies to include.
    scaled_theta : int
        Normalise library columns. 0 = no normalisation; 2 = L2 norm.
    ld : float
        Sequential-thresholding (STLS) parameter.
    gamma : float
        Tikhonov regularisation parameter (use 10**-inf for none).
    use_gls : float
        Generalised-least-squares weight (>0 enables GLS).
    """

    def __init__(self, polys=np.arange(0, 4), trigs=None,
                 scaled_theta=0, ld=0.05, gamma=10**(-np.inf), use_gls=1e-12):
        self.polys = np.asarray(polys)
        self.trigs = trigs if trigs is not None else []
        self.scale_theta = scaled_theta
        self.ld = ld
        self.gamma = gamma
        self.use_gls = use_gls
        self.coef = None
        self.tags = None
        self.feature_names_ = None

    # ── public fit methods ──────────────────────────────────────────────────

    def fit_uniform(self, X, t, L=30, overlap=0.5, custom_library=None):
        """Fit using a uniform test-function grid.

        Parameters
        ----------
        custom_library : CustomLibrary or None
            When provided, the library's feature_matrix is used in place of
            the internal monomial Theta.  Feature names are taken from the
            library's ``names`` attribute.  Pass ``None`` (default) to use
            the standard polynomial/trig library.
        """
        if custom_library is not None:
            Theta_0 = custom_library.feature_matrix(X)
            tags    = None          # tags are monomial-specific; unused for custom
            M_diag  = np.array([])
            self._custom_library = custom_library
        else:
            self._custom_library = None
            Theta_0, tags, M_diag = self._build_theta(X)

        n = X.shape[1]
        w = np.zeros((Theta_0.shape[1], n))
        V, Vp, grid = self._uniform_grid(t, L, overlap, [0, np.inf, 0])
        for i in range(n):
            G, b = self._compute_Gb(V, Vp, X, Theta_0, i)
            w_i = self._sparsify(G if self.scale_theta == 0 else G / M_diag.T,
                                 b, 1)
            if self.scale_theta > 0:
                w[:, i] = np.ndarray.flatten(w_i / M_diag)
            else:
                w[:, i] = np.ndarray.flatten(w_i)
        self.coef = w
        self.tags = tags
        if custom_library is not None:
            self.feature_names_ = list(custom_library.names)
        else:
            self._build_feature_names(X.shape[1])
        return self

    def fit_adaptive(self, X, t, r_whm=30, s=16, K=120, p=2, tau_p=16,
                     custom_library=None):
        """Fit using an adaptive (activity-based) test-function grid.

        Parameters
        ----------
        custom_library : CustomLibrary or None
            When provided, the library's feature_matrix is used in place of
            the internal monomial Theta.  Feature names are taken from the
            library's ``names`` attribute.  Pass ``None`` (default) to use
            the standard polynomial/trig library.
        """
        self._tau_p = tau_p
        if custom_library is not None:
            Theta_0 = custom_library.feature_matrix(X)
            tags    = None
            M_diag  = np.array([])
            self._custom_library = custom_library
        else:
            self._custom_library = None
            Theta_0, tags, M_diag = self._build_theta(X)

        n = X.shape[1]
        w = np.zeros((Theta_0.shape[1], n))
        wsindy_params = [s, K, p, 1]

        for i in range(n):
            grid_i = self._adaptive_grid(t, X[:, i], wsindy_params)
            V, Vp, _ = self._VVp_adaptive_whm(t, grid_i, r_whm, [0, np.inf, 0])
            G, b = self._compute_Gb(V, Vp, X, Theta_0, i)
            w_i = self._sparsify(G if self.scale_theta == 0 else G / M_diag.T,
                                 b, 1)
            if self.scale_theta > 0:
                w[:, i] = np.ndarray.flatten(w_i / M_diag)
            else:
                w[:, i] = np.ndarray.flatten(w_i)

        self.coef = w
        self.tags = tags
        if custom_library is not None:
            self.feature_names_ = list(custom_library.names)
        else:
            self._build_feature_names(X.shape[1])
        return self

    # ── predict / simulate ──────────────────────────────────────────────────

    def simulate(self, x0, t_span, t_eval):
        # self.tags  : (n_features, n_dims) — row 0 is all zeros (constant term)
        #              None when a custom library was used
        # self.coef  : (n_features, n_eqs)
        # We need  dx/dt = coef.T @ feature_vector
        coef_T         = self.coef.T                        # (n_eqs, n_features)
        custom_library = getattr(self, '_custom_library', None)

        if custom_library is not None:
            # Use the custom library's feature_vector for the RHS
            def rhs(t, x):
                return coef_T @ custom_library.feature_vector(np.asarray(x))
        else:
            tags = self.tags                                # (n_features, n_dims)
            def rhs(t, x):
                features = np.array([
                    np.prod([x[j] ** tags[i, j] for j in range(len(x))])
                    for i in range(len(tags))
                ])
                return coef_T @ features

        sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval,
                        method="RK45", rtol=1e-8, atol=1e-10)
        if not sol.success:
            raise RuntimeError(f"WSINDy simulation failed: {sol.message}")
        return sol.y.T

    def debug(self):
        """Print shapes and discovered equations for inspection."""
        print(f"  tags shape  : {self.tags.shape}  (n_features, n_dims)")
        print(f"  coef shape  : {self.coef.shape}  (n_features, n_eqs)")
        print(f"  feature names ({len(self.feature_names_)}): {self.feature_names_}")
        print("  Discovered equations (coef.T rows = equations):")
        for i, row in enumerate(self.coef.T):
            terms = [f"({c:+.4f}){n}" for c, n in
                     zip(row, self.feature_names_) if abs(c) > 1e-6]
            print(f"    eq{i}: {' '.join(terms) if terms else '0'}")

    def get_feature_names(self):
        return self.feature_names_

    def get_coefficients(self):
        """Return coefficient matrix (n_eqs, n_features)."""
        return self.coef.T   # (n_eqs, n_features)

    # ── internal helpers ────────────────────────────────────────────────────

    def _compute_Gb(self, V, Vp, X, Theta_0, i):
        if self.use_gls > 0:
            Cov = Vp @ Vp.T + self.use_gls * np.eye(V.shape[0])
            RT = np.linalg.cholesky(Cov)
            G = lstsq(RT, V @ Theta_0)[0]
            b = lstsq(RT, Vp @ X[:, i])[0]
        else:
            scale = 1.0 / np.linalg.norm(Vp, 2, axis=1)
            scale = scale.reshape(-1, 1)
            G = (V @ Theta_0) * scale
            b = scale.T * (Vp @ X[:, i])
        return G, b

    def _sparsify(self, Theta, dXdt, n, M=None):
        if M is None:
            M = np.ones((Theta.shape[1], 1))
        if self.gamma == 0:
            Theta_r = Theta
            dXdt_r  = dXdt.reshape(-1, 1)
        else:
            nn = Theta.shape[1]
            Theta_r  = np.vstack([Theta, self.gamma * np.eye(nn)])
            dXdt_col = dXdt.reshape(-1, 1)
            dXdt_r   = np.vstack([dXdt_col, self.gamma * np.zeros((nn, n))])

        Xi = M * lstsq(Theta_r, dXdt_r)[0]
        for _ in range(10):
            small = np.abs(Xi) < self.ld
            while np.sum(small) == Xi.size:
                self.ld /= 2
                small = np.abs(Xi) < self.ld
            Xi[small] = 0.0
            for ind in range(n):
                big = ~small[:, ind]
                if big.any():
                    rhs = dXdt_r[:, ind].reshape(-1, 1)
                    Xi[big, ind] = np.ndarray.flatten(
                        M[big] * lstsq(Theta_r[:, big], rhs)[0])
        return Xi

    def _build_theta(self, X):
        theta_0, tags = self._pool_data(X)
        if self.scale_theta > 0:
            M_diag = np.linalg.norm(theta_0, self.scale_theta, axis=0).reshape(-1, 1)
        else:
            M_diag = np.array([])
        return theta_0, tags, M_diag

    def _pool_data(self, X):
        n, d = X.shape
        P = int(self.polys[-1]) if len(self.polys) > 0 else 0
        rhs_functions = {}
        powers = []
        for p in range(1, P + 1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d - 1):
                starts = [0] + [idx + 1 for idx in indices]
                stops  = indices + (size,)
                pwr = tuple(map(operator.sub, stops, starts))
                powers.append(pwr)
        for pwr in powers:
            rhs_functions[pwr] = [lambda t, x=pwr: np.prod(np.power(list(t), list(x))), pwr]

        theta_0 = np.ones((n, 1))
        tags    = np.array(powers) if powers else np.zeros((0, d))

        for k in rhs_functions:
            func = rhs_functions[k][0]
            col  = np.array([func(X[i, :]) for i in range(n)]).reshape(-1, 1)
            theta_0 = np.hstack([theta_0, col])

        for freq in self.trigs:
            sin_col = np.sin(freq * X).sum(axis=1, keepdims=True)
            cos_col = np.cos(freq * X).sum(axis=1, keepdims=True)
            theta_0 = np.hstack([theta_0, sin_col, cos_col])
            trig_row = np.array([[-freq * 1j] * d, [freq * 1j] * d])
            tags = np.vstack([tags, trig_row])

        tags = np.vstack([np.zeros((1, d)), tags])
        return theta_0, tags

    def _build_feature_names(self, d):
        names = ["1"]
        var_names = ["x", "y", "z"][:d]
        P = int(self.polys[-1]) if len(self.polys) > 0 else 0
        for p in range(1, P + 1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d - 1):
                starts = [0] + [idx + 1 for idx in indices]
                stops  = indices + (size,)
                powers = tuple(map(operator.sub, stops, starts))
                term = "".join(
                    f"{var_names[i]}^{pwr}" if pwr > 1
                    else (var_names[i] if pwr == 1 else "")
                    for i, pwr in enumerate(powers)
                )
                names.append(term if term else "1")
        for freq in self.trigs:
            names += [f"sin({freq}·r)", f"cos({freq}·r)"]
        self.feature_names_ = names

    # ── test-function grids ─────────────────────────────────────────────────

    def _basis_fcn(self, p, q):
        def normalize(t, t1, tk):
            return (t - t1) ** max(p, 0) * (tk - t) ** max(q, 0)

        def g(t, t1, tk):
            num = ((p > 0) * (q > 0) * (t - t1)**max(p,0) * (tk - t)**max(q,0) +
                   (p == 0) * (q == 0) * (1 - 2*np.abs(t-(t1+tk)/2)/(tk-t1)) +
                   (p > 0) * (q < 0) * np.sin(p*np.pi/(tk-t1)*(t-t1)) +
                   (p == -1) * (q == -1))
            if p > 0 and q > 0:
                denom = np.abs(normalize((q*t1 + p*tk)/(p+q), t1, tk))
                return num / denom
            return num

        def gp(t, t1, tk):
            num = ((t-t1)**max(p-1,0) * (tk-t)**max(q-1,0) *
                   ((-p-q)*t + p*tk + q*t1) * (q > 0) * (p > 0) +
                   -2*np.sign(t-(t1+tk)/2)/(tk-t1) * (q == 0) * (p == 0) +
                   p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1)) * (q < 0) * (p > 0) +
                   0 * (p == -1) * (q == -1))
            if p > 0 and q > 0:
                denom = np.abs(normalize((q*t1 + p*tk)/(p+q), t1, tk))
                return num / denom
            return num

        return g, gp

    def _tf_mat_row(self, g, gp, t, t1, tk, param):
        N = len(t)
        pow_, nrm, ord_ = (param[0], param[1], param[2]) if param else (1, np.inf, 0)
        if t1 > tk:
            t1, tk = tk, t1
        V_row  = np.zeros((1, N))
        Vp_row = np.zeros((1, N))
        tg   = t[t1:tk+1]
        dts  = np.diff(tg)
        w    = 0.5 * (np.append(dts, [0]) + np.append([0], dts))
        V_row [:, t1:tk+1] = g (tg, t[t1], t[tk]) * w
        Vp_row[:, t1:tk+1] = -gp(tg, t[t1], t[tk]) * w
        Vp_row[:, t1]  -= g(t[t1], t[t1], t[tk])
        Vp_row[:, tk]  += g(t[tk], t[t1], t[tk])
        if pow_ != 0:
            sf = np.linalg.norm(np.ndarray.flatten(V_row[:, t1:tk+1]), nrm) if ord_ == 0 \
                 else np.linalg.norm(np.ndarray.flatten(Vp_row[:, t1:tk+1]), nrm)
            if sf > 0:
                V_row  /= sf
                Vp_row /= sf
        return V_row, Vp_row

    def _uniform_grid(self, t, L, s, param):
        M = len(t)
        p = 16
        overlap = int(np.floor(L * (1 - np.sqrt(1 - s**(1/p)))))
        grid, a, b = [], 0, int(L)
        grid.append([a, b])
        while b - overlap + L <= M - 1:
            a = b - overlap
            b = a + int(L)
            grid.append([a, b])
        grid = np.asarray(grid)
        N = len(grid)
        V = np.zeros((N, M));  Vp = np.zeros((N, M))
        g, gp = self._basis_fcn(p, p)
        for k in range(N):
            r, rp = self._tf_mat_row(g, gp, t, grid[k][0], grid[k][1], param)
            V[k, :] = r;  Vp[k, :] = rp
        return V, Vp, grid

    def _adaptive_grid(self, t, xobs, params):
        s, K, p, tau = params
        M = len(t)
        g, gp = self._basis_fcn(p, p)
        _, Vp_row = self._AG_tf_row(g, gp, t, 1, 1+s, [1, 1, 0])
        Vp_diags  = mb.repmat(Vp_row[:, 0:s+1], M-s, 1)
        Vp_sp     = scipy.sparse.diags(Vp_diags.T, np.arange(0, s+1), (M-s, M))
        weak_der  = Vp_sp.dot(xobs)
        pad = int(np.floor(s/2))
        weak_der = np.concatenate([np.zeros(pad), weak_der, np.zeros(pad)])
        Y = np.abs(weak_der);  Y = np.cumsum(Y);  Y /= Y[-1]
        Y = tau * Y + (1-tau) * np.linspace(Y[0], Y[-1], len(Y))
        t1 = Y[pad-1];  t2 = Y[len(Y)-int(np.ceil(s/2))-1]
        U  = np.linspace(t1, t2, K+2)
        grid = np.unique([np.argwhere(Y - U[i+1] >= 0)[0] for i in range(K)])
        return grid

    def _AG_tf_row(self, g, gp, t, t1, tk, param=None):
        N = len(t)
        gap, nrm, ord_ = (param[0], param[1], param[2]) if param else (1, np.inf, 0)
        if t1 > tk: t1, tk = tk, t1
        V_row  = np.zeros((1, N));  Vp_row = np.zeros((1, N))
        tg = t[t1:tk+1:gap];  dts = np.diff(tg)
        w  = 0.5 * (np.append(dts, [0]) + np.append([0], dts))
        V_row [:, t1:tk+1:gap] = g (tg, t[t1], t[tk]) * w
        Vp_row[:, t1:tk+1:gap] = -gp(tg, t[t1], t[tk]) * w
        Vp_row[:, t1] -= g(t[t1], t[t1], t[tk])
        Vp_row[:, tk] += g(t[tk], t[t1], t[tk])
        sf = np.linalg.norm(np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
        if sf > 0:
            V_row /= sf;  Vp_row /= sf
        return V_row, Vp_row

    def _test_fcn_param(self, r, c, t):
        tau_p = getattr(self, '_tau_p', 16)
        dt     = t[1] - t[0]
        r_whm  = r * dt
        A      = np.log2(10) * tau_p

        def ff(s):
            return (s - dt)**2 - (-s**2 * ((1 - (r_whm/s)**2)**A - 1))

        s = brentq(ff, r_whm, r_whm * np.sqrt(A) + dt)
        p = min(int(np.ceil(max(-1/np.log2(1 - (r_whm/s)**2), 1))), 200)
        a_arr = np.argwhere(t >= (c - s))
        a = int(a_arr[0]) if len(a_arr) else 0
        b = int(np.argwhere(t >= (c + s))[0]) if c + s <= t[-1] else len(t) - 1
        return p, a, b

    def _VVp_adaptive_whm(self, t, centers, r_whm, param):
        N = len(t);  M = len(centers)
        V = np.zeros((M, N));  Vp = np.zeros((M, N));  ab = np.zeros((M, 2))
        p, a, b = self._test_fcn_param(r_whm, t[int(centers[0])-1], t)
        a = int(a);  b = int(b)
        if b - a < 10:
            ctr = (a + b)//2;  a = max(0, ctr-5);  b = min(len(t)-1, ctr+5)
        g, gp = self._basis_fcn(p, p)
        Vr, Vpr = self._tf_mat_row(g, gp, t, a, b, param)
        V[0, :] = Vr;  Vp[0, :] = Vpr;  ab[0] = [a, b]
        for k in range(1, M):
            shift = int(centers[k] - centers[k-1])
            b_new = min(b + shift, len(t)-1)
            if a > 0 and b_new < len(t):
                a += shift;  b = b_new
                Vr  = np.roll(Vr,  shift)
                Vpr = np.roll(Vpr, shift)
            else:
                p, a, b = self._test_fcn_param(r_whm, t[int(centers[k])-1], t)
                a = int(a);  b = int(b)
                if b - a < 10:
                    ctr = (a+b)//2;  a = max(0, ctr-5);  b = min(len(t)-1, ctr+5)
                g, gp = self._basis_fcn(p, p)
                Vr, Vpr = self._tf_mat_row(g, gp, t, a, b, param)
            V[k, :] = Vr;  Vp[k, :] = Vpr;  ab[k] = [a, b]
        return V, Vp, ab