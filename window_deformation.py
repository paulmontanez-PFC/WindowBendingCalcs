"""Nonlinear deformation of a uniformly pressurized circular window.

This module solves the axisymmetric Foppl-von Karman large-deflection plate
equations for a circular window (e.g. an optical window) subjected to a uniform
pressure load on one face. It reports the transverse deflection profile, the
in-plane and bending stress distributions, the peak tensile stress, and the
safety factor against a user-supplied allowable stress.

Two edge support models are available:

* ``clamped``          - w(a) = 0, w'(a) = 0, u(a) = 0
* ``simply_supported`` - w(a) = 0, M_r(a) = 0, u(a) = 0 (radially constrained)

The public API is intentionally small and side-effect free so it can be reused
and unit tested independently of the command-line interface and plotting code.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_bvp

# --- Unit conversions -------------------------------------------------------
MM_PER_M = 1_000.0
PA_PER_MPA = 1.0e6
PA_PER_PSI = 6894.757293  # 1 psi = 6894.757293 Pa

# --- Solver configuration defaults ------------------------------------------
DEFAULT_MESH_NODES = 250
DEFAULT_OUTPUT_POINTS = 400
SOLVER_TOLERANCE = 1.0e-7
SOLVER_MAX_NODES = 50_000
# Small fraction of the radius used to avoid the 1/r singularity at the centre.
CENTER_REGULARIZATION = 1.0e-6
# The Mindlin P-form carries mild 1/r terms that need a milder centre cutoff.
MINDLIN_CENTER_REGULARIZATION = 1.0e-4

# --- Thin-plate solver robustness -------------------------------------------
# For very thin (membrane-dominated) plates the von Karman deflection grows to
# order the thickness (w0/t >~ 1); the clamped edge then develops a sharp
# bending boundary layer that a uniform-mesh collocation solver cannot resolve
# within its node budget. The following settings add a graceful recovery path.
#
# Number of load-continuation (homotopy) steps attempted when the single-shot
# Kirchhoff solve fails: the pressure is ramped from a small fraction to full,
# reusing each converged solution as the next initial guess.
KIRCHHOFF_CONTINUATION_STEPS = 10
# At or below this linear w0/t ratio the response is effectively linear, so the
# closed-form Kirchhoff solution is accurate (<~2%) and is returned as a
# graceful fallback when the nonlinear BVP will not converge.
LINEAR_FALLBACK_WT_RATIO = 0.2
# Above this w0/t ratio the Foppl-von Karman moderate-rotation assumption is no
# longer reliable; a warning tells the user the *model* (not just the solver)
# is outside its validity envelope and a large-rotation shell theory is needed.
VON_KARMAN_WT_WARN_RATIO = 5.0

# --- Figure output ----------------------------------------------------------
FIGURE_DIR = "figures"
FIGURE_FORMAT = "jpg"
FIGURE_DPI = 150

# --- Spreadsheet output -----------------------------------------------------
EXCEL_FORMAT = "xlsx"

# --- Default fused silica material properties -------------------------------
FUSED_SILICA_YOUNGS_MODULUS_PA = 72.0e9
FUSED_SILICA_POISSON_RATIO = 0.17
# Allowable design stress = modulus of rupture / 10 (680 psi for fused silica),
# per the reference glass-property tables. This is also the shared default.
FUSED_SILICA_ALLOWABLE_STRESS_PA = 680.0 * PA_PER_PSI

# --- Default window geometry / loading --------------------------------------
DEFAULT_DIAMETER_MM = 450.0
DEFAULT_THICKNESS_MM = 25.0
DEFAULT_PRESSURE_PA = 101_000.0

CLAMPED = "clamped"
SIMPLY_SUPPORTED = "simply_supported"
VALID_BOUNDARY_CONDITIONS = (CLAMPED, SIMPLY_SUPPORTED)

# --- Plate theory selection -------------------------------------------------
KIRCHHOFF = "kirchhoff"
MINDLIN = "mindlin"
AUTO = "auto"
VALID_PLATE_THEORIES = (KIRCHHOFF, MINDLIN, AUTO)
# Shear correction factor kappa for a homogeneous rectangular section (5/6).
SHEAR_CORRECTION_FACTOR = 5.0 / 6.0
# Thin-plate (Kirchhoff/FvK) validity limit on thickness/diameter. Above this
# ratio transverse shear is non-negligible, so ``auto`` switches to Mindlin.
# Radius-to-thickness (R/h) criteria for selecting plate theory, expressed via the
# thickness/diameter ratio t/D = h/D = 1 / (2 * R/h):
#   * R/h <= 5   (t/D >= 0.10): Mindlin-Reissner required (definite thick plate).
#   * 5 < R/h <= 10 (0.05 <= t/D < 0.10): transition zone -- Mindlin recommended.
#   * R/h > 10   (t/D < 0.05): Kirchhoff-Love thin-plate theory is valid.
# ``auto`` selects Mindlin whenever R/h <= 10 (t/D >= 0.05) so the transition zone
# uses the more accurate model, and Kirchhoff only for genuinely thin plates.
# R/h = 5 boundary: Mindlin definitely required.
MINDLIN_THICKNESS_RATIO_THRESHOLD = 0.1
# R/h = 10 boundary: ``auto`` switches to Mindlin here (inclusive), covering the
# transition zone; below this Kirchhoff is used.
MINDLIN_AUTO_SWITCH_RATIO = 0.05
# Relative tolerance so boundary cases (e.g. t/D exactly 0.05) are classified
# consistently despite floating-point rounding in the metre-based ratio.
MINDLIN_THRESHOLD_REL_TOL = 1.0e-9


def _auto_selects_mindlin(ratio: float) -> bool:
    """Return True when ``auto`` should use Mindlin for this t/D ratio.

    Mindlin is selected whenever R/h <= 10 (t/D >= ``MINDLIN_AUTO_SWITCH_RATIO``),
    i.e. the transition zone and thicker. The boundary at t/D = 0.05 is inclusive
    (transition -> Mindlin); ``math.isclose`` keeps that boundary stable against
    floating-point rounding in the metre-based ratio computation.
    """
    if math.isclose(
        ratio, MINDLIN_AUTO_SWITCH_RATIO, rel_tol=MINDLIN_THRESHOLD_REL_TOL
    ):
        return True
    return ratio > MINDLIN_AUTO_SWITCH_RATIO


def _display_plate_theory(theory: str) -> str:
    """Capitalized display label for a plate-theory identifier.

    The lowercase constants (``kirchhoff``/``mindlin``) are internal identifiers
    and CLI tokens; user-facing output shows them with a capitalized first
    letter (``Kirchhoff``/``Mindlin``). Other values (e.g. ``auto``) are returned
    unchanged.
    """
    if theory == KIRCHHOFF:
        return "Kirchhoff"
    if theory == MINDLIN:
        return "Mindlin"
    return theory


THICKNESS_SWEEP = "thickness_mm"
PRESSURE_SWEEP = "pressure_pa"
DIAMETER_SWEEP = "diameter_mm"
VALID_SWEEP_VARIABLES = (THICKNESS_SWEEP, PRESSURE_SWEEP, DIAMETER_SWEEP)

# Combined sweep: vary thickness and diameter together as a 2-D grid.
COMBINED_SWEEP = "all"

# Default bounds used when a diameter sweep is requested without explicit
# start/stop values: radius 125-225 mm in 25 mm increments (diameter 250-450 mm).
DEFAULT_DIAMETER_SWEEP_START_MM = 250.0
DEFAULT_DIAMETER_SWEEP_STOP_MM = 450.0
DEFAULT_DIAMETER_SWEEP_COUNT = 5

# Default bounds for the thickness leg of the combined sweep.
DEFAULT_THICKNESS_SWEEP_START_MM = 15.0
DEFAULT_THICKNESS_SWEEP_STOP_MM = 35.0
DEFAULT_THICKNESS_SWEEP_COUNT = 5


@dataclass(frozen=True)
class Material:
    """Linear-elastic, isotropic material properties."""

    youngs_modulus_pa: float = FUSED_SILICA_YOUNGS_MODULUS_PA
    poisson_ratio: float = FUSED_SILICA_POISSON_RATIO
    allowable_stress_pa: float = FUSED_SILICA_ALLOWABLE_STRESS_PA
    name: str = "Fused Silica"

    def __post_init__(self) -> None:
        if self.youngs_modulus_pa <= 0.0:
            raise ValueError("youngs_modulus_pa must be positive.")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in the open interval (-1, 0.5).")
        if self.allowable_stress_pa <= 0.0:
            raise ValueError("allowable_stress_pa must be positive.")


# Named material presets selectable by name (nominal room-temperature values).
# The allowable stress is the "allowable design stress" = modulus of rupture / 10
# taken from the reference glass-property tables. Verify against your own
# certified data before relying on these for design.
DEFAULT_MATERIAL_NAME = "fused_silica"

MATERIAL_PRESETS: dict[str, Material] = {
    "fused_silica": Material(
        youngs_modulus_pa=FUSED_SILICA_YOUNGS_MODULUS_PA,
        poisson_ratio=FUSED_SILICA_POISSON_RATIO,
        allowable_stress_pa=FUSED_SILICA_ALLOWABLE_STRESS_PA,
        name="Fused Silica",
    ),
    "n_bk7": Material(82.0e9, 0.206, 500.0 * PA_PER_PSI, "N-BK7"),
    "borofloat": Material(64.0e9, 0.20, 25.0 * PA_PER_MPA, "Borofloat 33"),
    "aluminosilicate": Material(
        12.4e6 * PA_PER_PSI, 0.26, 660.0 * PA_PER_PSI, "Aluminosilicate (1723)"
    ),
    "vycor": Material(10.0e6 * PA_PER_PSI, 0.19, 615.0 * PA_PER_PSI, "96% Silica (Vycor)"),
    "pyrex": Material(9.1e6 * PA_PER_PSI, 0.20, 610.0 * PA_PER_PSI, "Borosilicate (Pyrex)"),
    "plate_glass": Material(
        10.0e6 * PA_PER_PSI, 0.21, 650.0 * PA_PER_PSI, "Plate Glass (Herculite)"
    ),
    "lead_glass": Material(
        8.0e6 * PA_PER_PSI, 0.23, 500.0 * PA_PER_PSI, "High-Lead X-Ray Glass"
    ),
    "acrylic": Material(
        360.0e3 * PA_PER_PSI, 0.39, 920.0 * PA_PER_PSI, "Methyl Methacrylate"
    ),
    "sapphire": Material(345.0e9, 0.29, 350.0 * PA_PER_MPA, "Sapphire"),
    "zerodur": Material(90.3e9, 0.243, 57.0 * PA_PER_MPA, "Zerodur"),
    "caf2": Material(75.8e9, 0.26, 36.0 * PA_PER_MPA, "Calcium Fluoride"),
    "beryllium": Material(287.0e9, 0.032, 240.0 * PA_PER_MPA, "Beryllium"),
}


def format_material_presets() -> str:
    """Return a human-readable table of the available material presets."""
    lines = [
        f"{'Key':<16}{'Name':<26}{'E (GPa)':>9}{'nu':>7}{'Allow (MPa)':>13}",
    ]
    for key, mat in MATERIAL_PRESETS.items():
        lines.append(
            f"{key:<16}{mat.name:<26}"
            f"{mat.youngs_modulus_pa / 1.0e9:>9.1f}"
            f"{mat.poisson_ratio:>7.3f}"
            f"{mat.allowable_stress_pa / PA_PER_MPA:>13.2f}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class PlateConfig:
    """Geometry, loading, material and edge support for a circular plate."""

    diameter_m: float
    thickness_m: float
    pressure_pa: float
    material: Material = Material()
    boundary_condition: str = CLAMPED
    plate_theory: str = KIRCHHOFF

    def __post_init__(self) -> None:
        if self.diameter_m <= 0.0:
            raise ValueError("diameter_m must be positive.")
        if self.thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive.")
        if self.pressure_pa <= 0.0:
            raise ValueError("pressure_pa must be positive.")
        if self.boundary_condition not in VALID_BOUNDARY_CONDITIONS:
            raise ValueError(
                "boundary_condition must be one of "
                f"{VALID_BOUNDARY_CONDITIONS!r}, got {self.boundary_condition!r}."
            )
        if self.plate_theory not in VALID_PLATE_THEORIES:
            raise ValueError(
                "plate_theory must be one of "
                f"{VALID_PLATE_THEORIES!r}, got {self.plate_theory!r}."
            )

    @classmethod
    def from_engineering_units(
        cls,
        diameter_mm: float,
        thickness_mm: float,
        pressure_pa: float,
        material: Material | None = None,
        boundary_condition: str = CLAMPED,
        plate_theory: str = KIRCHHOFF,
    ) -> PlateConfig:
        """Build a config from millimetre geometry and an optional material."""
        return cls(
            diameter_m=diameter_mm / MM_PER_M,
            thickness_m=thickness_mm / MM_PER_M,
            pressure_pa=pressure_pa,
            material=material if material is not None else Material(),
            boundary_condition=boundary_condition,
            plate_theory=plate_theory,
        )

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    @property
    def thickness_to_diameter_ratio(self) -> float:
        """Thickness/diameter ratio used to gauge thin-plate validity."""
        return self.thickness_m / self.diameter_m

    @property
    def shear_modulus_pa(self) -> float:
        """Shear modulus G = E / (2 (1 + nu))."""
        nu = self.material.poisson_ratio
        return self.material.youngs_modulus_pa / (2.0 * (1.0 + nu))

    @property
    def shear_stiffness(self) -> float:
        """Transverse shear stiffness kappa * G * t (per unit length)."""
        return SHEAR_CORRECTION_FACTOR * self.shear_modulus_pa * self.thickness_m

    def resolved_plate_theory(self) -> str:
        """Return the concrete theory, resolving ``auto`` from the t/D ratio."""
        if self.plate_theory != AUTO:
            return self.plate_theory
        if _auto_selects_mindlin(self.thickness_to_diameter_ratio):
            return MINDLIN
        return KIRCHHOFF

    @property
    def flexural_rigidity(self) -> float:
        """Plate bending stiffness D = E t^3 / (12 (1 - nu^2))."""
        nu = self.material.poisson_ratio
        return self.material.youngs_modulus_pa * self.thickness_m**3 / (12.0 * (1.0 - nu**2))

    @property
    def membrane_stiffness(self) -> float:
        """In-plane stiffness C = E t / (1 - nu^2)."""
        nu = self.material.poisson_ratio
        return self.material.youngs_modulus_pa * self.thickness_m / (1.0 - nu**2)

    def linear_center_deflection(self) -> float:
        """Small-deflection (Kirchhoff) centre deflection for this support."""
        nu = self.material.poisson_ratio
        radius = self.radius_m
        if self.boundary_condition == CLAMPED:
            stiffness = 64.0 * self.flexural_rigidity / radius**4
        else:
            stiffness = (
                64.0 * self.flexural_rigidity * (1.0 + nu) / ((5.0 + nu) * radius**4)
            )
        return self.pressure_pa / stiffness


@dataclass(frozen=True)
class PlateSolution:
    """Computed deflection and stress fields for a solved plate."""

    r_m: np.ndarray
    w_m: np.ndarray
    w0_linear_m: float
    w0_nonlinear_m: float
    sigma_r_top_pa: np.ndarray
    sigma_t_top_pa: np.ndarray
    sigma_r_bottom_pa: np.ndarray
    sigma_t_bottom_pa: np.ndarray
    max_tensile_pa: float
    safety_factor: float
    boundary_condition: str
    plate_theory: str = KIRCHHOFF
    validity_warning: str | None = None

    @property
    def center_deflection_mm(self) -> float:
        return self.w0_nonlinear_m * MM_PER_M

    @property
    def max_tensile_mpa(self) -> float:
        return self.max_tensile_pa / PA_PER_MPA

    @property
    def exceeds_von_karman_validity(self) -> bool:
        """True when w0/t breaches the Foppl-von Karman validity limit."""
        return self.validity_warning is not None


def _radial_ratio(
    value: np.ndarray, center_limit: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Return ``value / r`` using the analytic limit at the centre node."""
    ratio = np.empty_like(value)
    ratio[0] = center_limit[0]
    ratio[1:] = value[1:] / r[1:]
    return ratio


def _make_ode(config: PlateConfig):
    """Create the first-order ODE system for the von Karman equations.

    The transverse equation is written using the Laplacian intermediate
    variable ``M = nabla^2 w`` so that only mild ``1/r`` singularities appear
    (the raw biharmonic form carries ``1/r^3`` terms that are poorly conditioned
    for the collocation solver). State vector::

        y = [w, w', M, M', u, u']

    where ``w`` is the transverse deflection, ``M`` the Laplacian of ``w`` and
    ``u`` the radial in-plane displacement.
    """
    nu = config.material.poisson_ratio
    pressure = config.pressure_pa
    d_bending = config.flexural_rigidity
    c_membrane = config.membrane_stiffness

    def ode(r: np.ndarray, y: np.ndarray) -> np.ndarray:
        w_r, m, m_r_deriv, u, u_r = y[1], y[2], y[3], y[4], y[5]

        w_r_over_r = w_r / r
        u_over_r = u / r
        w_rr = m - w_r_over_r

        n_r = c_membrane * (u_r + nu * u_over_r + 0.5 * w_r**2)
        n_t = c_membrane * (u_over_r + nu * u_r + 0.5 * nu * w_r**2)

        m_second = (pressure + n_r * w_rr + n_t * w_r_over_r) / d_bending - m_r_deriv / r
        u_second = (n_t - n_r) / (c_membrane * r) - nu * (u_r / r - u / r**2) - w_r * w_rr

        return np.vstack([w_r, w_rr, m_r_deriv, m_second, u_r, u_second])

    return ode


def _make_bc(config: PlateConfig):
    """Create the boundary-condition residual function for the BVP solver."""
    nu = config.material.poisson_ratio
    radius = config.radius_m
    boundary_condition = config.boundary_condition

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        if boundary_condition == CLAMPED:
            edge_rotation_or_moment = yb[1]  # w'(a) = 0
        else:
            # Radial moment M_r(a) = -D (w'' + nu w'/r) = 0, with w'' = M - w'/r
            # gives M(a) = (1 - nu) w'(a) / a.
            edge_rotation_or_moment = yb[2] - (1.0 - nu) * yb[1] / radius

        return np.array(
            [
                ya[1],  # w'(0)  = 0  (symmetry)
                ya[3],  # M'(0)  = 0  (symmetry)
                ya[4],  # u(0)   = 0  (symmetry)
                yb[0],  # w(a)   = 0  (supported edge)
                edge_rotation_or_moment,
                yb[4],  # u(a)   = 0  (radially constrained)
            ]
        )

    return bc


def _initial_guess(config: PlateConfig, r_mesh: np.ndarray) -> np.ndarray:
    """Clamped small-deflection shape used to seed the nonlinear solver."""
    radius = config.radius_m
    w0 = config.linear_center_deflection()
    x = r_mesh / radius

    w = w0 * (1.0 - x**2) ** 2
    w_r = -4.0 * w0 * r_mesh / radius**2 * (1.0 - x**2)
    m = -8.0 * w0 / radius**2 + 16.0 * w0 * r_mesh**2 / radius**4
    m_r_deriv = 32.0 * w0 * r_mesh / radius**4
    u = np.zeros_like(r_mesh)
    u_r = np.zeros_like(r_mesh)
    return np.vstack([w, w_r, m, m_r_deriv, u, u_r])


def _solve_mindlin_fields(
    config: PlateConfig, n_mesh: int, n_points: int
) -> tuple[np.ndarray, ...]:
    """Solve the axisymmetric Mindlin-Reissner (shear-deformable) plate BVP.

    Returns (r, w, M_r, M_t, N_r, N_t). The full Mindlin-von Karman system is
    solved for state ``[w, beta, P, Q, u, N_r]`` where ``beta`` is the
    independent section rotation, ``P = beta' + beta/r`` (a curvature-sum
    intermediate that keeps only mild ``1/r`` terms, mirroring the Kirchhoff
    solver), ``Q`` the transverse shear resultant and ``u`` the radial in-plane
    displacement. The problem is non-dimensionalised for conditioning; physical
    quantities are reconstructed inside the ODE.
    """
    radius = config.radius_m
    nu = config.material.poisson_ratio
    d_bending = config.flexural_rigidity
    c_membrane = config.membrane_stiffness
    et = config.material.youngs_modulus_pa * config.thickness_m
    ks = config.shear_stiffness
    pressure = config.pressure_pa

    w_scale = config.linear_center_deflection()
    if w_scale <= 0.0:
        w_scale = pressure * radius**4 / (64.0 * d_bending)
    scales = np.array(
        [
            w_scale,
            w_scale / radius,
            w_scale / radius**2,
            pressure * radius,
            w_scale**2 / radius,
            c_membrane * (w_scale / radius) ** 2,
        ]
    )
    inv_scale_times_a = radius / scales

    def ode(xi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = radius * xi
        _w, beta, p_curv, q, u, n_r = (y[i] * scales[i] for i in range(6))
        w_r = q / ks - beta
        n_t = nu * n_r + et * u / r
        beta_r = p_curv - beta / r
        p_curv_r = q / d_bending
        u_r = n_r / c_membrane - nu * u / r - 0.5 * w_r**2
        n_r_r = (n_t - n_r) / r
        q_r = (-q / r - n_t * w_r / r + n_r * beta_r - pressure) / (1.0 + n_r / ks)
        dphys = np.vstack([w_r, beta_r, p_curv_r, q_r, u_r, n_r_r])
        return dphys * inv_scale_times_a[:, None]

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        # Clamped: beta(a) = 0. Simply supported: M_r(a) = 0 (scaled form).
        edge = (
            yb[1]
            if config.boundary_condition == CLAMPED
            else yb[2] - (1.0 - nu) * yb[1]
        )
        return np.array(
            [
                ya[1],  # beta(0) = 0  (symmetry)
                ya[3],  # Q(0)    = 0  (symmetry)
                ya[4],  # u(0)    = 0  (symmetry)
                yb[0],  # w(a)    = 0  (supported edge)
                edge,
                yb[4],  # u(a)    = 0  (radially constrained)
            ]
        )

    xi0 = MINDLIN_CENTER_REGULARIZATION
    xi_mesh = np.linspace(xi0, 1.0, n_mesh)
    x = xi_mesh
    guess = np.vstack(
        [
            (1.0 - x**2) ** 2,
            4.0 * x * (1.0 - x**2),
            4.0 * (2.0 - 4.0 * x**2),
            -x / 2.0,
            np.zeros_like(x),
            np.zeros_like(x),
        ]
    )

    solution = solve_bvp(
        ode, bc, xi_mesh, guess, tol=SOLVER_TOLERANCE, max_nodes=SOLVER_MAX_NODES
    )
    if not solution.success:
        raise RuntimeError(f"Mindlin BVP solver failed: {solution.message}")

    r = np.linspace(0.0, radius, n_points)
    xi_eval = np.maximum(r / radius, solution.x[0])
    y = solution.sol(xi_eval)

    w = (y[0] * scales[0]).copy()
    beta = (y[1] * scales[1]).copy()
    p_curv = (y[2] * scales[2]).copy()
    u = (y[4] * scales[4]).copy()
    n_r = (y[5] * scales[5]).copy()

    beta[0] = 0.0
    u[0] = 0.0

    # beta/r -> P(0)/2 at the centre (beta ~ r near the origin).
    beta_over_r = np.empty_like(beta)
    beta_over_r[0] = p_curv[0] / 2.0
    beta_over_r[1:] = beta[1:] / r[1:]

    # u/r -> u'(0) = N_r(0) / (C (1 + nu)) at the centre (w'(0) = 0).
    u_over_r = np.empty_like(u)
    u_over_r[0] = n_r[0] / (c_membrane * (1.0 + nu))
    u_over_r[1:] = u[1:] / r[1:]

    m_r = d_bending * (p_curv - (1.0 - nu) * beta_over_r)
    m_t = d_bending * (nu * p_curv + (1.0 - nu) * beta_over_r)
    n_t = nu * n_r + et * u_over_r

    return r, w, m_r, m_t, n_r, n_t


def _linear_plate_fields(
    config: PlateConfig, r: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Closed-form small-deflection Kirchhoff fields; return (r, w, M_r, M_t, N_r, N_t).

    Used as an accurate graceful fallback in the near-linear thin-plate regime
    where the nonlinear BVP fails to converge but membrane stretching is
    negligible (so the linear solution is correct). Membrane resultants are
    identically zero in linear theory.
    """
    nu = config.material.poisson_ratio
    a = config.radius_m
    p = config.pressure_pa
    d_bending = config.flexural_rigidity
    a2 = a * a
    r2 = r * r

    if config.boundary_condition == CLAMPED:
        w = p / (64.0 * d_bending) * (a2 - r2) ** 2
        m_r = p / 16.0 * ((1.0 + nu) * a2 - (3.0 + nu) * r2)
        m_t = p / 16.0 * ((1.0 + nu) * a2 - (1.0 + 3.0 * nu) * r2)
    else:  # simply supported, radially constrained
        w = (
            p
            / (64.0 * d_bending)
            * (a2 - r2)
            * ((5.0 + nu) / (1.0 + nu) * a2 - r2)
        )
        m_r = p / 16.0 * (3.0 + nu) * (a2 - r2)
        m_t = p / 16.0 * ((3.0 + nu) * a2 - (1.0 + 3.0 * nu) * r2)

    zeros = np.zeros_like(r)
    return r, w, m_r, m_t, zeros, zeros


def _kirchhoff_fields_from_solution(
    config: PlateConfig, solution, n_points: int
) -> tuple[np.ndarray, ...]:
    """Sample a converged Kirchhoff BVP solution onto the output grid."""
    radius = config.radius_m
    nu = config.material.poisson_ratio
    d_bending = config.flexural_rigidity
    c_membrane = config.membrane_stiffness

    r = np.linspace(0.0, radius, n_points)
    r_eval = np.maximum(r, solution.x[0])
    y = solution.sol(r_eval)

    w = y[0].copy()
    w_r = y[1].copy()
    m = y[2].copy()
    u = y[4].copy()
    u_r = y[5].copy()

    # Enforce exact symmetry values at the centre node.
    w_r[0] = 0.0
    u[0] = 0.0

    # w'/r -> w''(0) = M(0)/2 in the limit at the centre.
    w_r_over_r = np.empty_like(w_r)
    w_r_over_r[0] = m[0] / 2.0
    w_r_over_r[1:] = w_r[1:] / r[1:]
    w_rr = m - w_r_over_r

    u_over_r = _radial_ratio(u, u_r, r)

    n_r = c_membrane * (u_r + nu * u_over_r + 0.5 * w_r**2)
    n_t = c_membrane * (u_over_r + nu * u_r + 0.5 * nu * w_r**2)

    m_r = -d_bending * (w_rr + nu * w_r_over_r)
    m_t = -d_bending * (w_r_over_r + nu * w_rr)

    return r, w, m_r, m_t, n_r, n_t


def _solve_kirchhoff_continuation(config: PlateConfig, n_mesh: int):
    """Retry the Kirchhoff BVP with load continuation; return solution or None.

    Ramps the pressure from a small fraction up to the full value, carrying the
    converged mesh and solution forward as the initial guess for each step. This
    reaches deeper into the nonlinear (membrane-influenced) regime than a single
    cold-started solve. Returns ``None`` if any step fails to converge.
    """
    radius = config.radius_m
    r_mesh = np.linspace(radius * CENTER_REGULARIZATION, radius, n_mesh)

    fractions = np.linspace(
        1.0 / KIRCHHOFF_CONTINUATION_STEPS, 1.0, KIRCHHOFF_CONTINUATION_STEPS
    )
    guess = _initial_guess(
        replace(config, pressure_pa=config.pressure_pa * fractions[0]), r_mesh
    )

    solution = None
    for frac in fractions:
        step_config = replace(config, pressure_pa=config.pressure_pa * frac)
        solution = solve_bvp(
            _make_ode(step_config),
            _make_bc(step_config),
            r_mesh,
            guess,
            tol=SOLVER_TOLERANCE,
            max_nodes=SOLVER_MAX_NODES,
        )
        if not solution.success:
            return None
        r_mesh, guess = solution.x, solution.y
    return solution


def _solve_kirchhoff_fields(
    config: PlateConfig, n_mesh: int, n_points: int
) -> tuple[np.ndarray, ...]:
    """Solve the Kirchhoff/von Karman BVP; return (r, w, M_r, M_t, N_r, N_t).

    A single-shot collocation solve is attempted first. If it fails (typically
    a thin, membrane-dominated plate with a sharp edge boundary layer), a
    load-continuation retry is made. If that also fails, the closed-form linear
    solution is returned when the response is effectively linear; otherwise an
    informative error is raised.
    """
    radius = config.radius_m
    r0 = radius * CENTER_REGULARIZATION
    r_mesh = np.linspace(r0, radius, n_mesh)

    solution = solve_bvp(
        _make_ode(config),
        _make_bc(config),
        r_mesh,
        _initial_guess(config, r_mesh),
        tol=SOLVER_TOLERANCE,
        max_nodes=SOLVER_MAX_NODES,
    )

    if not solution.success:
        solution = _solve_kirchhoff_continuation(config, n_mesh)

    if solution is None or not solution.success:
        wt_ratio = config.linear_center_deflection() / config.thickness_m
        if wt_ratio <= LINEAR_FALLBACK_WT_RATIO:
            warnings.warn(
                "von Karman BVP did not converge, but the deflection/thickness "
                f"ratio is small ({wt_ratio:.3f}); returning the accurate "
                "closed-form linear Kirchhoff solution instead.",
                stacklevel=2,
            )
            r = np.linspace(0.0, radius, n_points)
            return _linear_plate_fields(config, r)
        raise RuntimeError(
            "von Karman BVP failed to converge for this thin, large-deflection "
            f"plate (estimated deflection/thickness ~ {wt_ratio:.1f}). This is "
            "the strongly nonlinear, membrane-dominated regime where a sharp "
            "edge boundary layer forms; note that a ratio above "
            f"~{VON_KARMAN_WT_WARN_RATIO:.0f} also exceeds Foppl-von Karman "
            "validity and requires a large-rotation shell model."
        )

    return _kirchhoff_fields_from_solution(config, solution, n_points)


def _assemble_solution(
    config: PlateConfig,
    theory: str,
    r: np.ndarray,
    w: np.ndarray,
    m_r: np.ndarray,
    m_t: np.ndarray,
    n_r: np.ndarray,
    n_t: np.ndarray,
) -> PlateSolution:
    """Recover surface stresses and package a :class:`PlateSolution`."""
    thickness = config.thickness_m

    sigma_r_bending = 6.0 * m_r / thickness**2
    sigma_t_bending = 6.0 * m_t / thickness**2
    sigma_r_membrane = n_r / thickness
    sigma_t_membrane = n_t / thickness

    sigma_r_top = -sigma_r_bending + sigma_r_membrane
    sigma_t_top = -sigma_t_bending + sigma_t_membrane
    sigma_r_bottom = sigma_r_bending + sigma_r_membrane
    sigma_t_bottom = sigma_t_bending + sigma_t_membrane

    all_stresses = np.vstack([sigma_r_top, sigma_t_top, sigma_r_bottom, sigma_t_bottom])
    max_tensile = float(np.max(all_stresses))
    if max_tensile > 0.0:
        safety_factor = config.material.allowable_stress_pa / max_tensile
    else:
        safety_factor = float("inf")

    w0_nonlinear = float(np.max(w))
    validity_warning = _von_karman_validity_warning(w0_nonlinear / config.thickness_m)

    return PlateSolution(
        r_m=r,
        w_m=w,
        w0_linear_m=config.linear_center_deflection(),
        w0_nonlinear_m=w0_nonlinear,
        sigma_r_top_pa=sigma_r_top,
        sigma_t_top_pa=sigma_t_top,
        sigma_r_bottom_pa=sigma_r_bottom,
        sigma_t_bottom_pa=sigma_t_bottom,
        max_tensile_pa=max_tensile,
        safety_factor=safety_factor,
        boundary_condition=config.boundary_condition,
        plate_theory=theory,
        validity_warning=validity_warning,
    )


def _von_karman_validity_warning(w0_over_t: float) -> str | None:
    """Return a warning message if w0/t exceeds von Karman validity, else None."""
    if w0_over_t > VON_KARMAN_WT_WARN_RATIO:
        return (
            f"Central deflection/thickness = {w0_over_t:.1f} exceeds the "
            f"Foppl-von Karman moderate-rotation validity limit "
            f"(~{VON_KARMAN_WT_WARN_RATIO:.0f}); the small-strain/moderate-"
            "rotation assumption is violated and results may be inaccurate. A "
            "geometrically-exact large-rotation shell model is required here."
        )
    return None


def solve_plate(
    config: PlateConfig,
    n_mesh: int = DEFAULT_MESH_NODES,
    n_points: int = DEFAULT_OUTPUT_POINTS,
) -> PlateSolution:
    """Solve the plate problem with the configured (or auto-selected) theory.

    ``config.plate_theory`` selects Kirchhoff/von Karman (thin plate) or
    Mindlin-Reissner (shear-deformable). ``auto`` chooses Mindlin when the
    thickness/diameter ratio exceeds the thin-plate validity limit.
    """
    if n_mesh < 4:
        raise ValueError("n_mesh must be at least 4.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    theory = config.resolved_plate_theory()
    if theory == MINDLIN:
        fields = _solve_mindlin_fields(config, n_mesh, n_points)
    else:
        fields = _solve_kirchhoff_fields(config, n_mesh, n_points)
    solution = _assemble_solution(config, theory, *fields)

    if solution.validity_warning is not None:
        warnings.warn(solution.validity_warning, stacklevel=2)
    return solution


@dataclass(frozen=True)
class SweepPoint:
    """One evaluated case in a design sweep."""

    value: float
    center_deflection_mm: float
    peak_stress_mpa: float
    safety_factor: float
    exceeds_von_karman: bool = False


@dataclass(frozen=True)
class GridPoint:
    """One evaluated case in a two-parameter (thickness x diameter) grid sweep."""

    thickness_mm: float
    diameter_mm: float
    center_deflection_mm: float
    peak_stress_mpa: float
    safety_factor: float
    exceeds_von_karman: bool = False
    linear_center_deflection_mm: float = 0.0
    resolved_plate_theory: str = KIRCHHOFF

    @property
    def thickness_to_diameter_ratio(self) -> float:
        """Thickness/diameter ratio for this grid case."""
        return self.thickness_mm / self.diameter_mm


def run_sweep_grid(
    base_config: PlateConfig,
    thickness_values_mm: Sequence[float],
    diameter_values_mm: Sequence[float],
    n_points: int = DEFAULT_OUTPUT_POINTS,
) -> list[GridPoint]:
    """Evaluate the plate response over a full thickness x diameter grid.

    Pressure, material and boundary condition are taken from ``base_config``;
    thickness and diameter both vary. Returns structured data so callers can plot
    a family of curves or print matrices.
    """
    if len(thickness_values_mm) == 0 or len(diameter_values_mm) == 0:
        raise ValueError("Grid sweep requires at least one thickness and diameter.")

    points: list[GridPoint] = []
    for thickness_mm in thickness_values_mm:
        for diameter_mm in diameter_values_mm:
            case_config = PlateConfig(
                diameter_m=diameter_mm / MM_PER_M,
                thickness_m=thickness_mm / MM_PER_M,
                pressure_pa=base_config.pressure_pa,
                material=base_config.material,
                boundary_condition=base_config.boundary_condition,
                plate_theory=base_config.plate_theory,
            )
            solution = solve_plate(case_config, n_points=n_points)
            points.append(
                GridPoint(
                    thickness_mm=float(thickness_mm),
                    diameter_mm=float(diameter_mm),
                    center_deflection_mm=solution.center_deflection_mm,
                    peak_stress_mpa=solution.max_tensile_mpa,
                    safety_factor=solution.safety_factor,
                    exceeds_von_karman=solution.exceeds_von_karman_validity,
                    linear_center_deflection_mm=solution.w0_linear_m * MM_PER_M,
                    resolved_plate_theory=solution.plate_theory,
                )
            )
    return points


def run_design_sweep(
    base_config: PlateConfig,
    sweep_variable: str,
    sweep_values: Sequence[float],
    n_points: int = DEFAULT_OUTPUT_POINTS,
) -> list[SweepPoint]:
    """Evaluate the plate response across a range of one design variable.

    ``sweep_variable`` selects which quantity varies; all other parameters are
    taken from ``base_config``. Returns structured data so callers can print,
    plot or further analyse the results.
    """
    if sweep_variable not in VALID_SWEEP_VARIABLES:
        raise ValueError(
            f"sweep_variable must be one of {VALID_SWEEP_VARIABLES!r}, "
            f"got {sweep_variable!r}."
        )
    if len(sweep_values) == 0:
        raise ValueError("sweep_values must contain at least one value.")

    points: list[SweepPoint] = []
    for value in sweep_values:
        if sweep_variable == THICKNESS_SWEEP:
            case_config = PlateConfig(
                diameter_m=base_config.diameter_m,
                thickness_m=value / MM_PER_M,
                pressure_pa=base_config.pressure_pa,
                material=base_config.material,
                boundary_condition=base_config.boundary_condition,
                plate_theory=base_config.plate_theory,
            )
        elif sweep_variable == DIAMETER_SWEEP:
            case_config = PlateConfig(
                diameter_m=value / MM_PER_M,
                thickness_m=base_config.thickness_m,
                pressure_pa=base_config.pressure_pa,
                material=base_config.material,
                boundary_condition=base_config.boundary_condition,
                plate_theory=base_config.plate_theory,
            )
        else:
            case_config = PlateConfig(
                diameter_m=base_config.diameter_m,
                thickness_m=base_config.thickness_m,
                pressure_pa=value,
                material=base_config.material,
                boundary_condition=base_config.boundary_condition,
                plate_theory=base_config.plate_theory,
            )

        solution = solve_plate(case_config, n_points=n_points)
        points.append(
            SweepPoint(
                value=float(value),
                center_deflection_mm=solution.center_deflection_mm,
                peak_stress_mpa=solution.max_tensile_mpa,
                safety_factor=solution.safety_factor,
                exceeds_von_karman=solution.exceeds_von_karman_validity,
            )
        )
    return points


def _sweep_axis_label(sweep_variable: str) -> str:
    if sweep_variable == THICKNESS_SWEEP:
        return "Thickness (mm)"
    if sweep_variable == DIAMETER_SWEEP:
        return "Diameter (mm)"
    return "Pressure (Pa)"


def _held_constant_label(base_config: PlateConfig, sweep_variable: str) -> str:
    """Return a human-readable summary of the parameters kept fixed in a sweep."""
    parts: list[str] = []
    if sweep_variable != DIAMETER_SWEEP:
        parts.append(f"Diameter = {base_config.diameter_m * MM_PER_M:.1f} mm")
    if sweep_variable != THICKNESS_SWEEP:
        parts.append(f"Thickness = {base_config.thickness_m * MM_PER_M:.1f} mm")
    if sweep_variable != PRESSURE_SWEEP:
        parts.append(f"Pressure = {base_config.pressure_pa:,.0f} Pa")
    parts.append(f"Edge = {base_config.boundary_condition}")
    parts.append(f"Theory = {_display_plate_theory(base_config.resolved_plate_theory())}")
    return "Held constant: " + ", ".join(parts)


def _get_pyplot():
    """Return pyplot configured with the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _annotate_validity(fig, solution: PlateSolution) -> None:
    """Draw an unmissable red validity banner on a figure when w0/t is too large."""
    if not solution.exceeds_von_karman_validity:
        return
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5,
        0.02,
        "MODEL INVALID: deflection exceeds Foppl-von Karman validity "
        f"(w0/t > {VON_KARMAN_WT_WARN_RATIO:g}); results unreliable - "
        "use a large-rotation shell model.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="white",
        fontweight="bold",
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "red", "edgecolor": "black"},
    )


def _save_figure(fig, path: str) -> None:
    """Save a figure so titles, suptitles and banners are never clipped.

    ``bbox_inches="tight"`` expands the saved bounding box to enclose every
    artist, including a wide ``suptitle`` or the validity banner that would
    otherwise overflow the default figure bounds and be cut off.
    """
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)


def _unique_path(
    directory: str,
    name: str,
    extension: str,
    boundary_condition: str | None = None,
) -> str:
    """Return a path under ``directory`` that does not overwrite an existing file.

    When ``boundary_condition`` is given, it is always appended to the base name
    (``{name}_{boundary_condition}``). If that file already exists, a numeric
    suffix is added (``{name}_{boundary_condition}_1``, ``_2``, ...) until an
    unused name is found.
    """
    stem = f"{name}_{boundary_condition}" if boundary_condition else name
    candidate = os.path.join(directory, f"{stem}.{extension}")
    if not os.path.exists(candidate):
        return candidate

    index = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{index}.{extension}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _figure_path(
    name: str, output_dir: str, boundary_condition: str | None = None
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return _unique_path(output_dir, name, FIGURE_FORMAT, boundary_condition)


def _excel_path(
    name: str, output_dir: str, boundary_condition: str | None = None
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return _unique_path(output_dir, name, EXCEL_FORMAT, boundary_condition)


GRID_METRICS: tuple[tuple[str, Callable[[GridPoint], float]], ...] = (
    ("Center Deflection (mm)", lambda p: p.center_deflection_mm),
    ("Peak Tensile Stress (MPa)", lambda p: p.peak_stress_mpa),
    ("Safety Factor", lambda p: p.safety_factor),
)


def _autofit_columns(worksheet) -> None:
    """Widen each column to fit its longest cell value."""
    from openpyxl.utils import get_column_letter

    for column_cells in worksheet.columns:
        width = max(
            (len(str(cell.value)) for cell in column_cells if cell.value is not None),
            default=0,
        )
        letter = get_column_letter(column_cells[0].column)
        worksheet.column_dimensions[letter].width = width + 2


def _export_single_case_excel(
    config: PlateConfig,
    solution: PlateSolution,
    output_dir: str = FIGURE_DIR,
) -> list[str]:
    """Write the single-case summary to an Excel workbook."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Single Case"

    rows = [
        ("Quantity", "Value", "Units"),
        ("Diameter", config.diameter_m * MM_PER_M, "mm"),
        ("Thickness", config.thickness_m * MM_PER_M, "mm"),
        ("Pressure", config.pressure_pa, "Pa"),
        ("Boundary condition", config.boundary_condition, ""),
        ("Plate theory", _display_plate_theory(solution.plate_theory), ""),
        ("Thickness/diameter ratio", config.thickness_to_diameter_ratio, ""),
        ("Young's modulus", config.material.youngs_modulus_pa, "Pa"),
        ("Poisson's ratio", config.material.poisson_ratio, ""),
        ("Allowable stress", config.material.allowable_stress_pa / PA_PER_MPA, "MPa"),
        ("Linear center deflection", solution.w0_linear_m * MM_PER_M, "mm"),
        ("Nonlinear center deflection", solution.center_deflection_mm, "mm"),
        ("Peak tensile stress", solution.max_tensile_mpa, "MPa"),
        ("Safety factor", solution.safety_factor, ""),
    ]
    for row in rows:
        sheet.append(row)
    sheet["A1"].font = _bold_font()
    sheet["B1"].font = _bold_font()
    sheet["C1"].font = _bold_font()

    if solution.validity_warning is not None:
        _append_validity_row(sheet, solution.validity_warning)

    _append_theory_note(sheet)
    _autofit_columns(sheet)

    path = _excel_path("single_case", output_dir, config.boundary_condition)
    workbook.save(path)
    return [path]


def _export_sweep_excel(
    sweep_variable: str,
    points: Sequence[SweepPoint],
    output_dir: str = FIGURE_DIR,
    base_config: PlateConfig | None = None,
) -> list[str]:
    """Write a single-parameter sweep to an Excel workbook."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Design Sweep"

    if base_config is not None:
        sheet.append([_held_constant_label(base_config, sweep_variable)])
        sheet.append([])

    header = (
        _sweep_axis_label(sweep_variable),
        "Center Deflection (mm)",
        "Peak Tensile Stress (MPa)",
        "Safety Factor",
    )
    header_row = sheet.max_row + 1
    sheet.append(header)
    for cell in sheet[header_row]:
        cell.font = _bold_font()

    from openpyxl.styles import PatternFill

    red_fill = PatternFill(
        start_color="FFC00000", end_color="FFC00000", fill_type="solid"
    )
    any_flagged = False
    for point in points:
        sheet.append(
            [
                point.value,
                point.center_deflection_mm,
                point.peak_stress_mpa,
                point.safety_factor,
            ]
        )
        for column in (2, 3, 4):
            sheet.cell(row=sheet.max_row, column=column).number_format = "0.00000"
        if point.exceeds_von_karman:
            any_flagged = True
            sheet.cell(row=sheet.max_row, column=1).fill = red_fill

    if any_flagged:
        flagged = ", ".join(f"{p.value:g}" for p in points if p.exceeds_von_karman)
        _append_validity_row(
            sheet,
            "Row(s) highlighted red exceed Foppl-von Karman validity "
            f"(w0/t > {VON_KARMAN_WT_WARN_RATIO:g}); swept value(s): {flagged}. "
            "Those cases require a large-rotation shell model.",
        )
    _append_theory_note(sheet)
    _autofit_columns(sheet)

    bc = base_config.boundary_condition if base_config is not None else None
    path = _excel_path(f"sweep_{sweep_variable}", output_dir, bc)
    workbook.save(path)
    return [path]


def _held_constant_rows(base_config: PlateConfig) -> list[tuple]:
    """Return the constant load and material parameter rows for sweep sheets."""
    return [
        ("Pressure", base_config.pressure_pa, "Pa"),
        ("Boundary condition", base_config.boundary_condition, ""),
        ("Plate theory", _display_plate_theory(base_config.plate_theory), ""),
        ("Young's modulus", base_config.material.youngs_modulus_pa, "Pa"),
        ("Poisson's ratio", base_config.material.poisson_ratio, ""),
        (
            "Allowable stress",
            base_config.material.allowable_stress_pa / PA_PER_MPA,
            "MPa",
        ),
    ]


def _write_grid_details_sheet(
    sheet,
    points: Sequence[GridPoint],
    base_config: PlateConfig | None,
    flagged_points: Sequence[GridPoint],
    red_fill,
) -> None:
    """Populate the per-case "Case Details" worksheet for a combined sweep.

    Lists every grid case with the same set of quantities reported for a single
    case, so the combined workbook contains all of the single-case data.
    """
    sheet.title = "Case Details"

    if base_config is not None:
        sheet.append(["Parameters held constant"])
        sheet.cell(row=sheet.max_row, column=1).font = _bold_font()
        for row in _held_constant_rows(base_config):
            sheet.append(row)
            sheet.cell(row=sheet.max_row, column=1).font = _bold_font()
        sheet.append([])

    sheet.append(["Per-case results"])
    sheet.cell(row=sheet.max_row, column=1).font = _bold_font()

    header = (
        "Thickness (mm)",
        "Diameter (mm)",
        "Thickness/diameter ratio",
        "Plate theory",
        "Linear center deflection (mm)",
        "Nonlinear center deflection (mm)",
        "Peak tensile stress (MPa)",
        "Safety factor",
    )
    sheet.append(header)
    for cell in sheet[sheet.max_row]:
        cell.font = _bold_font()

    ordered = sorted(points, key=lambda p: (p.thickness_mm, p.diameter_mm))
    for point in ordered:
        sheet.append(
            [
                point.thickness_mm,
                point.diameter_mm,
                point.thickness_to_diameter_ratio,
                _display_plate_theory(point.resolved_plate_theory),
                point.linear_center_deflection_mm,
                point.center_deflection_mm,
                point.peak_stress_mpa,
                point.safety_factor,
            ]
        )
        data_row = sheet.max_row
        sheet.cell(row=data_row, column=3).number_format = "0.00000"
        for column in (5, 6, 7, 8):
            sheet.cell(row=data_row, column=column).number_format = "0.00000"
        if point.exceeds_von_karman:
            for column in range(1, len(header) + 1):
                sheet.cell(row=data_row, column=column).fill = red_fill

    if flagged_points:
        combos = ", ".join(
            f"(t={p.thickness_mm:g}, D={p.diameter_mm:g})" for p in flagged_points
        )
        _append_validity_row(
            sheet,
            "Row(s) highlighted red exceed Foppl-von Karman validity "
            f"(w0/t > {VON_KARMAN_WT_WARN_RATIO:g}); case(s): {combos}. "
            "Those require a large-rotation shell model.",
        )
    _append_theory_note(sheet)
    _autofit_columns(sheet)


def _export_grid_excel(
    points: Sequence[GridPoint],
    output_dir: str = FIGURE_DIR,
    base_config: PlateConfig | None = None,
) -> list[str]:
    """Write the combined thickness x diameter grid to an Excel workbook.

    The first worksheet ("Case Details") lists every grid case with the full set
    of per-case quantities also reported for a single case (geometry, ratio,
    resolved theory, linear and nonlinear deflection, peak stress and safety
    factor). Each response metric then gets its own matrix worksheet, laid out
    with thickness down the rows and diameter across the columns.
    """
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    thicknesses = sorted({p.thickness_mm for p in points})
    diameters = sorted({p.diameter_mm for p in points})
    lookup = {(p.thickness_mm, p.diameter_mm): p for p in points}
    red_fill = PatternFill(
        start_color="FFC00000", end_color="FFC00000", fill_type="solid"
    )
    flagged_points = [p for p in points if p.exceeds_von_karman]

    workbook = Workbook()
    details_sheet = workbook.active
    _write_grid_details_sheet(
        details_sheet, points, base_config, flagged_points, red_fill
    )

    for title, getter in GRID_METRICS:
        sheet = workbook.create_sheet()
        sheet.title = _sheet_title(title)

        if base_config is not None:
            sheet.append(["Parameters held constant"])
            sheet["A1"].font = _bold_font()
            for row in _held_constant_rows(base_config):
                sheet.append(row)
                sheet.cell(row=sheet.max_row, column=1).font = _bold_font()
            sheet.append([])
        sheet.append([f"Combined Sweep: {title}"])
        sheet.append([])

        header_row = ["Thickness (mm) \\ Diameter (mm)", *diameters]
        sheet.append(header_row)
        for cell in sheet[sheet.max_row]:
            cell.font = _bold_font()

        for thickness in thicknesses:
            row = [thickness]
            row.extend(getter(lookup[(thickness, d)]) for d in diameters)
            sheet.append(row)
            data_row = sheet.max_row
            sheet.cell(row=data_row, column=1).font = _bold_font()
            for offset, diameter in enumerate(diameters):
                column = 2 + offset
                sheet.cell(row=data_row, column=column).number_format = "0.00000"
                if lookup[(thickness, diameter)].exceeds_von_karman:
                    sheet.cell(row=data_row, column=column).fill = red_fill

        if flagged_points:
            combos = ", ".join(
                f"(t={p.thickness_mm:g}, D={p.diameter_mm:g})" for p in flagged_points
            )
            _append_validity_row(
                sheet,
                "Cell(s) highlighted red exceed Foppl-von Karman validity "
                f"(w0/t > {VON_KARMAN_WT_WARN_RATIO:g}); case(s): {combos}. "
                "Those require a large-rotation shell model.",
            )
        _autofit_columns(sheet)

    bc = base_config.boundary_condition if base_config is not None else None
    path = _excel_path("sweep_combined", output_dir, bc)
    workbook.save(path)
    return [path]


def _bold_font():
    from openpyxl.styles import Font

    return Font(bold=True)


def _append_theory_note(sheet) -> None:
    """Append an italic note explaining the t/D plate-theory selection rule."""
    from openpyxl.styles import Font

    thin_limit = MINDLIN_AUTO_SWITCH_RATIO
    thick_limit = MINDLIN_THICKNESS_RATIO_THRESHOLD
    sheet.append([])
    sheet.append(
        [
            "Note",
            "Plate theory selection (auto), by radius/thickness R/h = 1/(2 t/D): "
            f"Kirchhoff thin-plate theory is used for t/D < {thin_limit:g} "
            "(R/h > 10); Mindlin-Reissner thick-plate theory is used for "
            f"t/D >= {thin_limit:g} (R/h <= 10). This spans the transition zone "
            f"({thin_limit:g} <= t/D < {thick_limit:g}, i.e. 5 < R/h <= 10, where "
            f"Mindlin is recommended) and the definite thick-plate region "
            f"t/D >= {thick_limit:g} (R/h <= 5). An explicit --plate-theory "
            "kirchhoff or mindlin choice overrides this rule.",
        ]
    )
    row = sheet.max_row
    note_font = Font(italic=True)
    for column in (1, 2):
        sheet.cell(row=row, column=column).font = note_font


def _append_validity_row(sheet, message: str) -> None:
    """Append a highlighted red validity-warning row to an Excel worksheet."""
    from openpyxl.styles import Font, PatternFill

    sheet.append([])
    sheet.append(["!!! MODEL VALIDITY WARNING !!!", message])
    row = sheet.max_row
    red_fill = PatternFill(start_color="FFC00000", end_color="FFC00000", fill_type="solid")
    white_bold = Font(bold=True, color="FFFFFFFF")
    for column in (1, 2):
        cell = sheet.cell(row=row, column=column)
        cell.fill = red_fill
        cell.font = white_bold


def _sheet_title(metric_title: str) -> str:
    """Return an Excel-safe worksheet name (<=31 chars, no illegal characters)."""
    cleaned = metric_title.split(" (")[0]
    for bad in r"[]:*?/\\":
        cleaned = cleaned.replace(bad, "")
    return cleaned[:31]


def _case_parameter_label(config: PlateConfig) -> str:
    """Return a summary of the fixed input parameters for a single-case plot.

    The label is wrapped onto two lines so that, even for the longest
    boundary-condition and theory names, the subtitle stays within a sensible
    figure width instead of overflowing the plot edges.
    """
    return (
        f"Diameter = {config.diameter_m * MM_PER_M:.1f} mm, "
        f"Thickness = {config.thickness_m * MM_PER_M:.1f} mm, "
        f"Pressure = {config.pressure_pa:,.0f} Pa,\n"
        f"Edge = {config.boundary_condition}, "
        f"Theory = {_display_plate_theory(config.resolved_plate_theory())} "
        f"(t/D = {config.thickness_to_diameter_ratio:.3f})"
    )


def _plot_single_case(
    solution: PlateSolution,
    output_dir: str = FIGURE_DIR,
    config: PlateConfig | None = None,
) -> list[str]:
    plt = _get_pyplot()

    r_mm = solution.r_m * MM_PER_M
    w_mm = solution.w_m * MM_PER_M
    parameter_label = _case_parameter_label(config) if config is not None else None
    bc = config.boundary_condition if config is not None else None

    deflection_path = _figure_path("single_case_deflection", output_dir, bc)
    plt.figure(figsize=(7, 4.5))
    plt.plot(r_mm, w_mm, linewidth=2)
    plt.xlabel("Radius (mm)")
    plt.ylabel("Deflection (mm)")
    plt.title("Deflection Profile")
    plt.grid(True, alpha=0.3)
    if parameter_label is not None:
        plt.suptitle(parameter_label, fontsize=9)
    _annotate_validity(plt.gcf(), solution)
    plt.tight_layout()
    _save_figure(plt.gcf(), deflection_path)
    plt.close()

    stress_path = _figure_path("single_case_stress", output_dir, bc)
    plt.figure(figsize=(7, 4.5))
    plt.plot(r_mm, solution.sigma_r_top_pa / PA_PER_MPA, label=r"$\sigma_r$ top")
    plt.plot(r_mm, solution.sigma_t_top_pa / PA_PER_MPA, label=r"$\sigma_\theta$ top")
    plt.plot(
        r_mm,
        solution.sigma_r_bottom_pa / PA_PER_MPA,
        "--",
        label=r"$\sigma_r$ bottom",
    )
    plt.plot(
        r_mm,
        solution.sigma_t_bottom_pa / PA_PER_MPA,
        "--",
        label=r"$\sigma_\theta$ bottom",
    )
    plt.xlabel("Radius (mm)")
    plt.ylabel("Stress (MPa)")
    plt.title("Stress Profiles")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    if parameter_label is not None:
        plt.suptitle(parameter_label, fontsize=9)
    _annotate_validity(plt.gcf(), solution)
    plt.tight_layout()
    _save_figure(plt.gcf(), stress_path)
    plt.close()

    return [deflection_path, stress_path]


def _plot_sweep(
    sweep_variable: str,
    points: Sequence[SweepPoint],
    output_dir: str = FIGURE_DIR,
    base_config: PlateConfig | None = None,
) -> list[str]:
    plt = _get_pyplot()

    values = [p.value for p in points]
    deflection = [p.center_deflection_mm for p in points]
    stress = [p.peak_stress_mpa for p in points]
    safety = [p.safety_factor for p in points]

    x_label = _sweep_axis_label(sweep_variable)
    bc = base_config.boundary_condition if base_config is not None else None
    figure_path = _figure_path(f"sweep_{sweep_variable}", output_dir, bc)

    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(values, deflection, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Center deflection (mm)")
    plt.title("Deflection vs Sweep Variable")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(values, stress, marker="o", label="Peak stress (MPa)")
    plt.plot(values, safety, marker="^", label="Safety factor")
    plt.xlabel(x_label)
    plt.title("Stress and Safety vs Sweep Variable")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if base_config is not None:
        plt.suptitle(_held_constant_label(base_config, sweep_variable), fontsize=10)

    plt.tight_layout()
    _save_figure(plt.gcf(), figure_path)
    plt.close()

    return [figure_path]


def _plot_combined_sweep(
    points: Sequence[GridPoint],
    thickness_values_mm: Sequence[float],
    diameter_values_mm: Sequence[float],
    output_dir: str = FIGURE_DIR,
    base_config: PlateConfig | None = None,
) -> list[str]:
    """Plot a thickness x diameter grid sweep as families of curves.

    Top row: response versus thickness, one curve per diameter.
    Bottom row: response versus diameter, one curve per thickness.
    Columns: center deflection, peak stress, safety factor.
    """
    plt = _get_pyplot()

    bc = base_config.boundary_condition if base_config is not None else None
    figure_path = _figure_path("sweep_combined", output_dir, bc)
    lookup = {(p.thickness_mm, p.diameter_mm): p for p in points}
    thicknesses = sorted({p.thickness_mm for p in points})
    diameters = sorted({p.diameter_mm for p in points})

    metrics = (
        ("Center deflection (mm)", lambda p: p.center_deflection_mm),
        ("Peak stress (MPa)", lambda p: p.peak_stress_mpa),
        ("Safety factor", lambda p: p.safety_factor),
    )

    fig, axes = plt.subplots(2, len(metrics), figsize=(15, 8), squeeze=False)

    for col, (ylabel, getter) in enumerate(metrics):
        ax = axes[0][col]
        for diameter in diameters:
            y_values = [getter(lookup[(t, diameter)]) for t in thicknesses]
            ax.plot(thicknesses, y_values, marker="o", label=f"{diameter:.0f} mm")
        ax.set_xlabel("Thickness (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Thickness")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Diameter", fontsize=8)

    for col, (ylabel, getter) in enumerate(metrics):
        ax = axes[1][col]
        for thickness in thicknesses:
            y_values = [getter(lookup[(thickness, d)]) for d in diameters]
            ax.plot(diameters, y_values, marker="^", label=f"{thickness:.0f} mm")
        ax.set_xlabel("Diameter (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Diameter")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Thickness", fontsize=8)

    if base_config is not None:
        fig.suptitle(
            f"Held constant: Pressure = {base_config.pressure_pa:,.0f} Pa, "
            f"Edge = {base_config.boundary_condition}, "
            f"Theory = {_display_plate_theory(base_config.plate_theory)}",
            fontsize=12,
        )

    fig.tight_layout()
    _save_figure(fig, figure_path)
    plt.close(fig)

    return [figure_path]


def _print_validity_banner(message: str) -> None:
    """Print an unmissable boxed banner for a model-validity breach to stderr."""
    import sys

    wrapped: list[str] = []
    for paragraph in message.split("\n"):
        line = ""
        for word in paragraph.split():
            if line and len(line) + 1 + len(word) > 72:
                wrapped.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        wrapped.append(line)

    width = max(len(line) for line in wrapped)
    width = max(width, len("!!! MODEL VALIDITY WARNING !!!"))
    border = "*" * (width + 6)
    print(border, file=sys.stderr)
    print("*  " + "!!! MODEL VALIDITY WARNING !!!".ljust(width) + "  *", file=sys.stderr)
    print("*  " + " " * width + "  *", file=sys.stderr)
    for line in wrapped:
        print("*  " + line.ljust(width) + "  *", file=sys.stderr)
    print(border, file=sys.stderr)


def _plate_theory_label(theory: str) -> str:
    """Human-readable name for a resolved plate theory."""
    if theory == MINDLIN:
        return "Mindlin-Reissner (shear-deformable) Plate BVP"
    return "von Karman Nonlinear (thin) Plate BVP"


def _print_single_case(config: PlateConfig, solution: PlateSolution) -> None:
    if solution.validity_warning is not None:
        _print_validity_banner(solution.validity_warning)
    print(
        f"=== {config.material.name} Window Deformation "
        f"({_plate_theory_label(solution.plate_theory)}) ==="
    )
    print(
        f"Diameter: {config.diameter_m * MM_PER_M:.3f} mm, "
        f"Thickness: {config.thickness_m * MM_PER_M:.3f} mm, "
        f"Pressure: {config.pressure_pa:.3f} Pa"
    )
    print(f"Boundary condition: {config.boundary_condition}")
    ratio = config.thickness_to_diameter_ratio
    requested = config.plate_theory
    resolved = solution.plate_theory
    if requested == resolved:
        theory_note = _display_plate_theory(resolved)
    else:
        theory_note = (
            f"{_display_plate_theory(requested)} -> {_display_plate_theory(resolved)}"
        )
    print(f"Plate theory: {theory_note} (t/D = {ratio:.3f})")
    if resolved == KIRCHHOFF and _auto_selects_mindlin(ratio):
        print(
            "  WARNING: t/D is at or above the thin-plate limit "
            f"({MINDLIN_AUTO_SWITCH_RATIO:g}; R/h <= 10); Kirchhoff may be "
            "inaccurate here -- consider --plate-theory mindlin."
        )
    print()
    print(f"Linear center deflection:    {solution.w0_linear_m * MM_PER_M:.4f} mm")
    print(f"Nonlinear center deflection: {solution.center_deflection_mm:.4f} mm")
    print(f"Peak tensile stress:         {solution.max_tensile_mpa:.3f} MPa")
    print(f"Safety factor (allowable/peak): {solution.safety_factor:.3f}")
    if solution.validity_warning is not None:
        print()
        _print_validity_banner(solution.validity_warning)


def _print_sweep(sweep_variable: str, points: Sequence[SweepPoint]) -> None:
    print("=== Design Sweep Summary ===")
    print(f"Sweep variable: {_sweep_axis_label(sweep_variable)}")
    print()
    header = f"{'Value':>10}  {'Deflection(mm)':>16}  {'PeakStress(MPa)':>16}  {'SafetyFactor':>14}"
    print(header)
    print("-" * len(header))
    for point in points:
        flag = "  <-- EXCEEDS von Karman" if point.exceeds_von_karman else ""
        print(
            f"{point.value:>10.3f}  {point.center_deflection_mm:>16.5f}  "
            f"{point.peak_stress_mpa:>16.5f}  {point.safety_factor:>14.4f}{flag}"
        )
    flagged = [p.value for p in points if p.exceeds_von_karman]
    if flagged:
        values = ", ".join(f"{v:g}" for v in flagged)
        _print_validity_banner(
            "The following swept value(s) exceed Foppl-von Karman validity "
            f"(w0/t > {VON_KARMAN_WT_WARN_RATIO:g}) and are marked above: "
            f"{values}. Results for those cases require a large-rotation shell "
            "model and should not be trusted."
        )


def _print_grid(
    points: Sequence[GridPoint],
    metric_getter: Callable[[GridPoint], float],
    title: str,
) -> None:
    """Print a thickness x diameter matrix for one response metric."""
    thicknesses = sorted({p.thickness_mm for p in points})
    diameters = sorted({p.diameter_mm for p in points})
    lookup = {(p.thickness_mm, p.diameter_mm): p for p in points}

    print(f"=== Combined Sweep: {title} ===")
    print("Rows: Thickness (mm)  |  Columns: Diameter (mm)")
    corner = "Thk\\Dia"
    header = f"{corner:>10}" + "".join(f"{d:>14.1f}" for d in diameters)
    print(header)
    print("-" * len(header))
    for thickness in thicknesses:
        row = f"{thickness:>10.1f}" + "".join(
            f"{metric_getter(lookup[(thickness, d)]):>14.5f}" for d in diameters
        )
        print(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nonlinear circular window deformation solver (von Karman BVP)."
    )
    parser.add_argument("--diameter-mm", type=float, default=DEFAULT_DIAMETER_MM)
    parser.add_argument("--thickness-mm", type=float, default=DEFAULT_THICKNESS_MM)
    parser.add_argument("--pressure-pa", type=float, default=DEFAULT_PRESSURE_PA)
    parser.add_argument(
        "--material",
        choices=list(MATERIAL_PRESETS),
        default=DEFAULT_MATERIAL_NAME,
        help=(
            "Named material preset. Sets Young's modulus, Poisson's ratio and "
            "allowable stress; individual --youngs-modulus-pa/--poisson-ratio/"
            "--allowable-stress-mpa flags override the preset value. Use "
            "--list-materials to print the preset table."
        ),
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="Print the available material presets and exit.",
    )
    parser.add_argument(
        "--youngs-modulus-pa",
        type=float,
        default=None,
        help="Override the preset Young's modulus (Pa).",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=None,
        help="Override the preset Poisson's ratio.",
    )
    parser.add_argument(
        "--allowable-stress-mpa",
        type=float,
        default=None,
        help="Override the preset allowable stress (MPa).",
    )
    parser.add_argument(
        "--boundary-condition",
        choices=list(VALID_BOUNDARY_CONDITIONS),
        default=CLAMPED,
        help="Edge support model.",
    )
    parser.add_argument(
        "--plate-theory",
        choices=list(VALID_PLATE_THEORIES),
        default=AUTO,
        help=(
            "Plate theory: 'kirchhoff' (thin-plate von Karman), 'mindlin' "
            "(shear-deformable), or 'auto' (Mindlin when R/h <= 10, i.e. "
            f"t/D >= {MINDLIN_AUTO_SWITCH_RATIO:g}, else Kirchhoff)."
        ),
    )
    parser.add_argument(
        "--sweep-variable",
        choices=["none", *VALID_SWEEP_VARIABLES, COMBINED_SWEEP],
        default="none",
        help=(
            "Optional design sweep variable. Use 'all' to combine the thickness "
            "and diameter sweeps into one chart (at the fixed pressure)."
        ),
    )
    parser.add_argument("--sweep-start", type=float, default=None)
    parser.add_argument("--sweep-stop", type=float, default=None)
    parser.add_argument("--sweep-count", type=int, default=7)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing all output files (both figures and spreadsheets).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip writing JPEG figure files (spreadsheets are still written).",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip writing Excel spreadsheet files (figures are still written).",
    )
    parser.add_argument(
        "--output-dir",
        default=FIGURE_DIR,
        help="Directory for saved figures and spreadsheets.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PlateConfig:
    preset = MATERIAL_PRESETS[args.material]
    material = Material(
        youngs_modulus_pa=(
            args.youngs_modulus_pa
            if args.youngs_modulus_pa is not None
            else preset.youngs_modulus_pa
        ),
        poisson_ratio=(
            args.poisson_ratio
            if args.poisson_ratio is not None
            else preset.poisson_ratio
        ),
        allowable_stress_pa=(
            args.allowable_stress_mpa * PA_PER_MPA
            if args.allowable_stress_mpa is not None
            else preset.allowable_stress_pa
        ),
        name=preset.name,
    )
    return PlateConfig.from_engineering_units(
        diameter_mm=args.diameter_mm,
        thickness_mm=args.thickness_mm,
        pressure_pa=args.pressure_pa,
        material=material,
        boundary_condition=args.boundary_condition,
        plate_theory=args.plate_theory,
    )


def _default_sweep_bounds(sweep_variable: str) -> tuple[float, float, int]:
    """Return the default (start, stop, count) for a single-parameter sweep."""
    if sweep_variable == DIAMETER_SWEEP:
        return (
            DEFAULT_DIAMETER_SWEEP_START_MM,
            DEFAULT_DIAMETER_SWEEP_STOP_MM,
            DEFAULT_DIAMETER_SWEEP_COUNT,
        )
    if sweep_variable == THICKNESS_SWEEP:
        return (
            DEFAULT_THICKNESS_SWEEP_START_MM,
            DEFAULT_THICKNESS_SWEEP_STOP_MM,
            DEFAULT_THICKNESS_SWEEP_COUNT,
        )
    raise ValueError(f"No default bounds defined for sweep variable {sweep_variable!r}.")


def _run_combined_sweep(
    base_config: PlateConfig,
    save_figures: bool,
    save_excel: bool,
    output_dir: str,
) -> None:
    """Run the thickness x diameter grid, print matrices, and save outputs."""
    t_start, t_stop, t_count = _default_sweep_bounds(THICKNESS_SWEEP)
    d_start, d_stop, d_count = _default_sweep_bounds(DIAMETER_SWEEP)
    thickness_values = np.linspace(t_start, t_stop, t_count)
    diameter_values = np.linspace(d_start, d_stop, d_count)

    points = run_sweep_grid(base_config, thickness_values, diameter_values)

    _print_grid(points, lambda p: p.center_deflection_mm, "Center Deflection (mm)")
    print()
    _print_grid(points, lambda p: p.peak_stress_mpa, "Peak Tensile Stress (MPa)")
    print()
    _print_grid(points, lambda p: p.safety_factor, "Safety Factor")

    flagged = [p for p in points if p.exceeds_von_karman]
    if flagged:
        combos = ", ".join(
            f"(t={p.thickness_mm:g} mm, D={p.diameter_mm:g} mm)" for p in flagged
        )
        _print_validity_banner(
            "The following grid case(s) exceed Foppl-von Karman validity "
            f"(w0/t > {VON_KARMAN_WT_WARN_RATIO:g}): {combos}. Results for those "
            "cases require a large-rotation shell model and should not be trusted."
        )

    if save_figures:
        figures = _plot_combined_sweep(
            points, thickness_values, diameter_values, output_dir, base_config
        )
        for path in figures:
            print(f"Saved figure: {path}")
    if save_excel:
        for path in _export_grid_excel(points, output_dir, base_config):
            print(f"Saved spreadsheet: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.list_materials:
        print("Available material presets:")
        print(format_material_presets())
        return

    base_config = config_from_args(args)

    save_figures = not (args.no_save or args.no_figures)
    save_excel = not (args.no_save or args.no_excel)

    if args.sweep_variable == COMBINED_SWEEP:
        _run_combined_sweep(base_config, save_figures, save_excel, args.output_dir)
        return

    if args.sweep_variable != "none":
        start, stop, count = args.sweep_start, args.sweep_stop, args.sweep_count
        if args.sweep_variable == DIAMETER_SWEEP and start is None and stop is None:
            start, stop, count = _default_sweep_bounds(DIAMETER_SWEEP)
        if start is None or stop is None:
            raise ValueError("For a design sweep, provide --sweep-start and --sweep-stop.")
        if count < 1:
            raise ValueError("--sweep-count must be at least 1.")
        sweep_values = np.linspace(start, stop, count)
        points = run_design_sweep(base_config, args.sweep_variable, sweep_values)
        _print_sweep(args.sweep_variable, points)
        if save_figures:
            for path in _plot_sweep(
                args.sweep_variable, points, args.output_dir, base_config
            ):
                print(f"Saved figure: {path}")
        if save_excel:
            for path in _export_sweep_excel(
                args.sweep_variable, points, args.output_dir, base_config
            ):
                print(f"Saved spreadsheet: {path}")
        return

    solution = solve_plate(base_config, n_points=500)
    _print_single_case(base_config, solution)
    if save_figures:
        for path in _plot_single_case(solution, args.output_dir, base_config):
            print(f"Saved figure: {path}")
    if save_excel:
        for path in _export_single_case_excel(base_config, solution, args.output_dir):
            print(f"Saved spreadsheet: {path}")


if __name__ == "__main__":
    import sys

    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
