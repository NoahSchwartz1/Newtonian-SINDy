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
# Registry
# ─────────────────────────────────────────────────────────────────────────────

SYSTEMS: Dict[str, ODESystem] = {
    "lorenz":            LORENZ,
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
