"""
ode_systems.py
══════════════
Catalogue of ODE systems for the SINDy / WSINDy / GS-SINDy notebooks.

Every system is an ODESystem dataclass containing:
  • rhs            – callable (t, state, **params) → list[float]
  • default_params – dict of default parameter values
  • var_names      – list of state-variable names, e.g. ['x', 'y', 'z']
  • true_coeffs    – optional list of dicts (one per equation) mapping
                     feature-name string → true coefficient value.
                     Set to None when ground truth is unavailable.
  • description    – human-readable one-liner
  • custom_library – optional CustomLibrary for non-polynomial feature sets
                     (see CustomLibrary below). When set, the notebooks
                     automatically use it instead of a plain PolynomialLibrary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO USE CustomLibrary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A CustomLibrary lets you describe an *arbitrary* candidate function library
(rational terms, inverse powers, transcendentals, …) without modifying any
notebook.  The library is defined by two aligned lists:

  functions  – a list of callables, each mapping a state vector
               (1-D numpy array) → scalar.
  names      – a matching list of human-readable strings that identify each
               function in printed output and coefficient plots.

Example — 2-body planar orbit with state [x, y, vx, vy]:

    from ode_systems import CustomLibrary

    def inv_r3_x(s): x, y, vx, vy = s; r = np.hypot(x, y); return x / r**3
    def inv_r3_y(s): x, y, vx, vy = s; r = np.hypot(x, y); return y / r**3

    my_lib = CustomLibrary(
        functions=[lambda s: s[2],          # vx
                   lambda s: s[3],          # vy
                   inv_r3_x,               # x/r³
                   inv_r3_y],              # y/r³
        names=["vx", "vy", "x/r^3", "y/r^3"],
    )

The CustomLibrary converts to a PySINDy-compatible library automatically
via .to_pysindy_library() — call that when you need to pass it to ps.SINDy.

For WSINDy the CustomLibrary provides .feature_matrix(X) which evaluates
all functions on every row of X and returns an (N, n_features) array —
identical in shape to WSINDy's existing Theta matrix.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ADD A NEW SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  Write a plain RHS function:

        def my_rhs(t, state, param1=1.0, param2=0.5):
            x, y = state
            return [param1 * y, -param2 * x]

2.  (Optional) define a CustomLibrary if polynomial terms are insufficient:

        my_lib = CustomLibrary(
            functions=[lambda s: s[0], lambda s: np.sin(s[1])],
            names=["x", "sin(y)"],
        )

3.  Wrap it in ODESystem and register it:

        SYSTEMS["my_system"] = ODESystem(
            rhs=my_rhs,
            default_params=dict(param1=1.0, param2=0.5),
            var_names=["x", "y"],
            description="My 2-D system",
            custom_library=my_lib,   # omit for polynomial-only systems
            true_coeffs=[            # omit or set None if unknown
                {"y":  1.0},         # ẋ = y
                {"x": -0.5},         # ẏ = -0.5 x
            ],
        )

4.  In any notebook, set SYSTEM_KEY = "my_system".
    Everything else (dimension, variable names, metrics) adapts automatically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Currently registered systems
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lorenz            – Lorenz attractor (3-D)
  two_body_planar   – Newtonian 2-body problem, planar (4-D)
  three_body_planar – Newtonian 3-body problem, planar (12-D)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CustomLibrary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CustomLibrary:
    """
    A user-defined candidate function library.

    Parameters
    ----------
    functions : list of callables
        Each callable takes a 1-D numpy array (one state snapshot) and returns
        a scalar.  Vectorisation over the time axis is handled internally.
    names : list of str
        Human-readable label for each function.  Must be the same length as
        `functions`.  These strings are used in coefficient plots and printed
        equation output, and must exactly match the keys in `true_coeffs` if
        ground truth is provided.

    Notes
    -----
    • Functions are evaluated independently, so they can be fully arbitrary
      (rational terms, inverse powers, absolute values, etc.).
    • The library does *not* automatically include a constant/bias term.
      Add ``lambda s: 1.0`` (name ``"1"``) explicitly if you need it.
    • For PySINDy compatibility, each function must be safe to call on a
      1-D array.  Avoid functions that depend on array shape in unexpected ways.
    """

    functions: List[Callable]   # [f(state_1d) → scalar, …]
    names:     List[str]        # matching human-readable labels

    def __post_init__(self):
        if len(self.functions) != len(self.names):
            raise ValueError(
                f"CustomLibrary: `functions` and `names` must have the same "
                f"length, got {len(self.functions)} vs {len(self.names)}."
            )

    # ── evaluation ────────────────────────────────────────────────────────────

    def feature_matrix(self, X: np.ndarray) -> np.ndarray:
        """Evaluate every function on every row of X.

        Parameters
        ----------
        X : (N, n_dim) array — rows are state snapshots

        Returns
        -------
        Theta : (N, n_features) array
            The library / design matrix, identical in role to the Theta matrix
            used by WSINDy.
        """
        N = X.shape[0]
        Theta = np.empty((N, len(self.functions)))
        for j, fn in enumerate(self.functions):
            for i in range(N):
                Theta[i, j] = fn(X[i])
        return Theta

    def feature_vector(self, x: np.ndarray) -> np.ndarray:
        """Evaluate every function on a single state snapshot x (1-D array).

        Useful for building the RHS callable required by
        ``ode_utils.simulate_from_coefficients``.
        """
        return np.array([fn(x) for fn in self.functions])

    # ── PySINDy compatibility ─────────────────────────────────────────────────

    def to_pysindy_library(self):
        """Return a ``pysindy.CustomLibrary`` wrapping these functions.

        The returned object can be passed directly as the
        ``feature_library`` argument to ``ps.SINDy()``.

        Raises
        ------
        ImportError
            If PySINDy is not installed (harmless for WSINDy-only workflows).
        """
        try:
            import pysindy as ps
        except ImportError as exc:
            raise ImportError(
                "PySINDy is required for to_pysindy_library(). "
                "Install it with:  pip install pysindy"
            ) from exc

        # PySINDy's CustomLibrary expects functions of individual variables,
        # not of the full state vector.  We wrap each function using a helper
        # that reconstructs the state from *args.
        def _wrap(fn):
            def wrapped(*args):
                state = np.asarray(args)
                return fn(state)
            return wrapped

        lib = ps.CustomLibrary(
            library_functions=[_wrap(fn) for fn in self.functions],
            function_names=[lambda *args, n=name: n for name in self.names],
        )
        return lib

    # ── introspection ─────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.functions)

    def __repr__(self):
        return (f"CustomLibrary(n_features={len(self)}, "
                f"names={self.names})")


# ─────────────────────────────────────────────────────────────────────────────
# ODESystem dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ODESystem:
    rhs:            Callable                          # f(t, state, **params) → list
    default_params: Dict                              # forwarded as kwargs to rhs
    var_names:      List[str]                         # one entry per state dimension
    description:    str = ""
    true_coeffs:    Optional[List[Dict[str, float]]] = None   # None if unknown
    custom_library: Optional[CustomLibrary]          = None   # None → use polynomial

    @property
    def n_dim(self) -> int:
        return len(self.var_names)

    def __call__(self, t, state):
        """Make the system directly callable as f(t, state)."""
        return self.rhs(t, state, **self.default_params)


# ─────────────────────────────────────────────────────────────────────────────
# Lorenz attractor (3-D)
#   ẋ = σ(y − x)
#   ẏ = x(ρ − z) − y
#   ż = xy − βz
# ─────────────────────────────────────────────────────────────────────────────

def _lorenz_rhs(t, s, sigma=10.0, beta=8/3, rho=28.0):
    x, y, z = s
    return [sigma*(y - x),
            x*(rho - z) - y,
            x*y - beta*z]

LORENZ = ODESystem(
    rhs=_lorenz_rhs,
    default_params=dict(sigma=10.0, beta=8/3, rho=28.0),
    var_names=["x", "y", "z"],
    description="Lorenz attractor  ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz",
    # Keys must match the feature-name strings produced by the chosen library.
    # PySINDy PolynomialLibrary(degree=2) with feature_names=['x','y','z']
    # produces: '1', 'x', 'y', 'z', 'x^2', 'x y', 'x z', 'y^2', 'y z', 'z^2'
    true_coeffs=[
        {"x": -10.0, "y":  10.0              },   # ẋ
        {"x":  28.0, "y":  -1.0, "x z": -1.0},   # ẏ
        {"x y": 1.0, "z": -8/3               },   # ż
    ],
    custom_library=None,  # polynomial library is sufficient for Lorenz
)


# ─────────────────────────────────────────────────────────────────────────────
# 2-body planar problem (4-D)
#   state = [x, y, vx, vy]
#
#   ẋ  = vx
#   ẏ  = vy
#   v̇x = −μ · x / r³       r = √(x² + y²)
#   v̇y = −μ · y / r³
#
# The true equations are linear in the library features
#   { vx, vy, x/r³, y/r³ }
# so SINDy can recover exact coefficients from clean data.
# ─────────────────────────────────────────────────────────────────────────────

def _two_body_rhs(t, s, mu=1.0):
    x, y, vx, vy = s
    r3 = (x**2 + y**2) ** 1.5
    return [vx,
            vy,
            -mu * x / r3,
            -mu * y / r3]

def _make_two_body_library(mu=1.0) -> CustomLibrary:
    """
    Candidate library for the planar 2-body problem.
    State ordering: [x, y, vx, vy]

    Features
    --------
    vx, vy           — velocity components (kinematic terms)
    x/r^3, y/r^3     — gravitational acceleration directions

    This is the *minimal exact* library: every true term appears exactly once,
    so the recovered coefficient matrix should be the identity (up to μ).

    To run a broader search (e.g. if μ is unknown), you can extend the list
    with additional candidate terms such as:
        lambda s: 1.0                           # constant bias
        lambda s: s[0]                          # x
        lambda s: s[0] / np.hypot(s[0],s[1])   # x/r  (1/r² gravity)
        lambda s: s[0] / np.hypot(s[0],s[1])**5 # x/r⁵ (higher-order)
    and append their names accordingly.
    """
    def _vx(s):       return s[2]
    def _vy(s):       return s[3]
    def _x_over_r3(s): r3 = (s[0]**2 + s[1]**2)**1.5; return s[0] / r3
    def _y_over_r3(s): r3 = (s[0]**2 + s[1]**2)**1.5; return s[1] / r3

    return CustomLibrary(
        functions=[_vx, _vy, _x_over_r3, _y_over_r3],
        names=["vx", "vy", "x/r^3", "y/r^3"],
    )

_TWO_BODY_MU = 1.0

TWO_BODY_PLANAR = ODESystem(
    rhs=_two_body_rhs,
    default_params=dict(mu=_TWO_BODY_MU),
    var_names=["x", "y", "vx", "vy"],
    description="Planar 2-body  ẋ=vx, ẏ=vy, v̇x=-μx/r³, v̇y=-μy/r³",
    custom_library=_make_two_body_library(mu=_TWO_BODY_MU),
    # true_coeffs keys must match CustomLibrary.names exactly.
    true_coeffs=[
        {"vx":  1.0                   },   # ẋ  = vx
        {"vy":  1.0                   },   # ẏ  = vy
        {"x/r^3": -_TWO_BODY_MU       },   # v̇x = -μ x/r³
        {"y/r^3": -_TWO_BODY_MU       },   # v̇y = -μ y/r³
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# 3-body planar problem (12-D)
#   Bodies 1, 2, 3 each with equal mass m.
#   state = [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]
#
#   Kinematics:  ẋᵢ = vxᵢ,  ẏᵢ = vyᵢ
#   Dynamics:    v̇xᵢ = Σⱼ≠ᵢ  Gm(xⱼ−xᵢ)/rᵢⱼ³
#                v̇yᵢ = Σⱼ≠ᵢ  Gm(yⱼ−yᵢ)/rᵢⱼ³
#
# true_coeffs is None — the 3-body problem has no closed-form analytical
# solution, and the coefficient representation in any fixed library is not
# sparse.  SINDy will attempt discovery but ground-truth comparison is
# disabled.  Trajectory and library evaluation metrics still work.
# ─────────────────────────────────────────────────────────────────────────────

def _three_body_rhs(t, s, G=1.0, m=1.0):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s

    def _acc(xi, yi, xj, yj, xk, yk):
        """Gravitational acceleration on body i from bodies j and k."""
        rij3 = ((xj-xi)**2 + (yj-yi)**2) ** 1.5 + 1e-30   # softening for stability
        rik3 = ((xk-xi)**2 + (yk-yi)**2) ** 1.5 + 1e-30
        ax = G*m*(xj-xi)/rij3 + G*m*(xk-xi)/rik3
        ay = G*m*(yj-yi)/rij3 + G*m*(yk-yi)/rik3
        return ax, ay

    ax1, ay1 = _acc(x1, y1, x2, y2, x3, y3)
    ax2, ay2 = _acc(x2, y2, x1, y1, x3, y3)
    ax3, ay3 = _acc(x3, y3, x1, y1, x2, y2)

    return [vx1, vy1, vx2, vy2, vx3, vy3,
            ax1, ay1, ax2, ay2, ax3, ay3]

def _make_three_body_library(G=1.0, m=1.0) -> CustomLibrary:
    """
    Candidate library for the planar 3-body problem.
    State ordering: [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]

    Features
    --------
    Kinematic terms (vxᵢ, vyᵢ for i=1,2,3):
        These feed directly into the position derivatives ẋᵢ = vxᵢ.

    Gravitational terms for each ordered pair (i, j):
        (xⱼ−xᵢ)/rᵢⱼ³  and  (yⱼ−yᵢ)/rᵢⱼ³

    In total: 6 kinematic + 12 gravitational = 18 features.

    Notes
    -----
    • The softening length (1e-30) used in the RHS is NOT applied here —
      library features use the exact 1/r³ form.  This is intentional: for
      well-separated bodies the softening is negligible, and keeping the
      library exact helps coefficient recovery.  If your trajectories pass
      very close together, add a small ε² inside the hypot call.
    • You can add higher-order terms (1/r², 1/r⁵) or polynomial combinations
      to this list to widen the search space.
    """
    # ── kinematic features ────────────────────────────────────────────────────
    kin_fns   = [lambda s, i=i: s[6 + i] for i in range(6)]
    kin_names = ["vx1","vy1","vx2","vy2","vx3","vy3"]

    # ── gravitational features (xⱼ−xᵢ)/rᵢⱼ³  and  (yⱼ−yᵢ)/rᵢⱼ³ ──────────
    # Body index → (x_idx, y_idx) in state vector
    pos = {1: (0,1), 2: (2,3), 3: (4,5)}
    grav_fns, grav_names = [], []
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            if i == j:
                continue
            xi_idx, yi_idx = pos[i]
            xj_idx, yj_idx = pos[j]

            def _dx_over_r3(s, xi=xi_idx, yi=yi_idx, xj=xj_idx, yj=yj_idx):
                dx = s[xj] - s[xi]; dy = s[yj] - s[yi]
                r3 = (dx**2 + dy**2) ** 1.5
                return dx / r3

            def _dy_over_r3(s, xi=xi_idx, yi=yi_idx, xj=xj_idx, yj=yj_idx):
                dx = s[xj] - s[xi]; dy = s[yj] - s[yi]
                r3 = (dx**2 + dy**2) ** 1.5
                return dy / r3

            grav_fns.extend([_dx_over_r3, _dy_over_r3])
            grav_names.extend([f"(x{j}-x{i})/r{i}{j}^3",
                                f"(y{j}-y{i})/r{i}{j}^3"])

    return CustomLibrary(
        functions=kin_fns   + grav_fns,
        names=    kin_names + grav_names,
    )

_THREE_BODY_G = 1.0
_THREE_BODY_M = 1.0

THREE_BODY_PLANAR = ODESystem(
    rhs=_three_body_rhs,
    default_params=dict(G=_THREE_BODY_G, m=_THREE_BODY_M),
    var_names=["x1","y1","x2","y2","x3","y3",
               "vx1","vy1","vx2","vy2","vx3","vy3"],
    description="Planar 3-body (equal masses)  12-D gravitational ODE",
    custom_library=_make_three_body_library(G=_THREE_BODY_G, m=_THREE_BODY_M),
    true_coeffs=None,   # no sparse closed-form representation
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

SYSTEMS: Dict[str, ODESystem] = {
    "lorenz":            LORENZ,
    "two_body_planar":   TWO_BODY_PLANAR,
    "three_body_planar": THREE_BODY_PLANAR,
    # Add further systems here as needed.
}


def list_systems() -> None:
    """Print a short summary of every registered system."""
    print(f"\n{'Key':<22}  {'Dim':>4}  {'Custom lib?':>11}  Description")
    print("─" * 72)
    for key, sys in SYSTEMS.items():
        has_lib = "yes" if sys.custom_library is not None else "—"
        print(f"{key:<22}  {sys.n_dim:>4}  {has_lib:>11}  {sys.description}")
    print()