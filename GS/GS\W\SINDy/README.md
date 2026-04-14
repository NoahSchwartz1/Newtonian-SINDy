# SINDy / WSINDy / GS-SINDy — ODE System Identification

Tested on the Lorenz system. Infrastructure is general: switching to a
different system requires only changing SYSTEM_KEY and X0 in the USER
PARAMETERS cell of each notebook.

## Files

| File | Purpose |
|------|---------|
| `01_SINDy.ipynb` | Standard SINDy via PySINDy (STLSQ) |
| `02_WSINDy.ipynb` | Weak form SINDy (no pointwise derivatives) |
| `03_GSSINDy.ipynb` | Group Sparse (GS) SINDy (multiple trajectories jointly) |
| `00_Main_Comparison.ipynb` | Runs all three, unified plots and noise sweep |
| `ode_systems.py` | ODE catalogue — **add new systems here** |
| `ode_utils.py` | Shared helpers: data generation, metrics, plotting |
| `wsindy_core.py` | WSINDy implementation (MathBioCU/PyWSINDy_ODE) |

## Requirements

    pip install pysindy numpy scipy matplotlib pynumdiff

GS-SINDy also needs the repo cloned next to the notebooks:

    git clone https://github.com/lindliu/GS-SINDy

If absent, the notebook falls back to a vairant of PySINDy.

## Run order

    01_SINDy  →  02_WSINDy  →  03_GSSINDy  →  00_Main_Comparison
    or 00_Main_Comparison

The main notebook refits everything inline when RUN_INLINE = True,
or loads pre-saved .pkl files when RUN_INLINE = False. 
01, 02 and 03 Scripts generatre .pkl files automatically.

## Switching systems

In every notebook the first USER PARAMETERS cell has:

    SYSTEM_KEY = 'lorenz'
    SYSTEM     = SYSTEMS[SYSTEM_KEY]
    X0         = [-8.0, 8.0, 27.0]   # must match SYSTEM.n_dim

Change SYSTEM_KEY and X0 to use a different system.
Add an ODESystem object to use a new system.

## Adding a new system

Edit ode_systems.py. Add a RHS function and an ODESystem entry:

    def my_rhs(t, state, param=1.0):
        x, y = state
        return [y, -param * x]

    SYSTEMS["my_system"] = ODESystem(
        rhs=my_rhs,
        default_params=dict(param=1.0),
        var_names=["x", "y"],
        description="My 2-D system",
        true_coeffs=[          # optional; omit or set None if unknown
            {"y":  1.0},       # equation 0: x-dot = y
            {"x": -1.0},       # equation 1: y-dot = -x
        ],
    )

The true_coeffs keys must match the feature-name strings that your
chosen library produces (e.g. PySINDy PolynomialLibrary degree-2 with
vars ['x','y'] gives: '1', 'x', 'y', 'x^2', 'x y', 'y^2').
Set true_coeffs=None if you don't have ground truth — metrics that
require it will be skipped automatically.

## Key parameters

### 01_SINDy.ipynb
- THRESHOLD     sparsity threshold (STLSQ)
- ALPHA         ridge regularisation
- POLY_DEGREE   polynomial library degree
- DIFF_METHOD   derivative method: 'spline' | 'smoothed_finite_difference' | 'finite_difference'
- CUSTOM_LIBRARY_FN   set to a callable returning a PySINDy library, or None

### 02_WSINDy.ipynb
- LD            sparsity threshold
- GAMMA         Tikhonov regularisation
- POLY_DEGREE   polynomial library degree
- GRID_TYPE     'uniform' or 'adaptive'
- L_UNIFORM     window width in timesteps (uniform grid)

### 03_GSSINDy.ipynb
- THRESHOLD_SINDY      sparsity threshold
- THRESHOLD_GROUP      group-sparsity cutoff
- THRESHOLD_SIMILARITY Wasserstein distance for 'same structure' detection
- NUM_SERIES    number of sub-windows per trajectory
- WINDOW_PER    fractional window length

