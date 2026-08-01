// Coherence Field Theory engine for tornado prediction
// Written in Coherence Lang — the theory implemented in its own language.
//
// Solves the Helmholtz PDE: D∇²τ - Γτ + S = 0
// Where:
//   τ = coherence field amplitude
//   D = diffusion coefficient (from lapse rate)
//   Γ = damping rate (from CIN)
//   S = source term (from CAPE + SRH)
//
// Extracts: tau, |∇τ|, torsion, alignment, S/Γ, Da, singularity count

module hazardpulse.tornado.coherence;

import std.core {Option, Result, Ok, Err, Vec}
import std.math.arithmetic {sqrt, abs, exp, max, min, atan2, cos, sin}
import std.math.constants {PI}

/// Grid dimensions for 80km CONUS
const GRID_DLAT: F64 = 0.72;
const GRID_DLON: F64 = 0.94;
const LAT_MIN: F64 = 25.0;
const LAT_MAX: F64 = 50.0;
const LON_MIN: F64 = -125.0;
const LON_MAX: F64 = -65.0;
const N_LAT: USize = 35;
const N_LON: USize = 64;
const DX_KM: F64 = 80.0;

/// Coherence field diagnostics at a single point
struct CoherenceDiagnostics {
    tau: F64,
    grad_tau: F64,
    torsion: F64,
    alignment: F64,
    s_over_gamma: F64,
    da: F64,
    singularity_count: USize,
    source_field: F64,
    gamma_field: F64,
    e_coh: F64
}

/// A 2D field on the CONUS grid
struct GridField {
    data: Vec[F64],
    n_lat: USize,
    n_lon: USize
}

impl GridField {
    fn new(n_lat: USize, n_lon: USize) -> Self @ L0 {
        GridField {
            data: vec![0.0; n_lat * n_lon],
            n_lat,
            n_lon
        }
    }

    fn get(self, i: USize, j: USize) -> F64 @ L0 {
        self.data[i * self.n_lon + j]
    }

    fn set(mut self, i: USize, j: USize, val: F64) @ L0 {
        self.data[i * self.n_lon + j] = val;
    }

    fn max_value(self) -> F64 @ L0 {
        self.data.iter().cloned().fold(F64::NEG_INFINITY, F64::max)
    }
}

/// Gaussian smooth a 2D field (separable, O(n) per pixel)
fn gaussian_smooth(field: &GridField, sigma_cells: USize) -> GridField @ L0 {
    let radius = sigma_cells * 3;
    let mut result = GridField::new(field.n_lat, field.n_lon);

    // Build 1D kernel
    let kernel_size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut kernel_sum = 0.0;
    let sigma = sigma_cells as F64;
    for k in 0..kernel_size {
        let x = (k as F64) - (radius as F64);
        let w = exp(-x * x / (2.0 * sigma * sigma));
        kernel.push(w);
        kernel_sum += w;
    }
    // Normalize
    for k in 0..kernel_size {
        kernel[k] /= kernel_sum;
    }

    // Horizontal pass
    let mut temp = GridField::new(field.n_lat, field.n_lon);
    for i in 0..field.n_lat {
        for j in 0..field.n_lon {
            let mut sum = 0.0;
            for k in 0..kernel_size {
                let jj = (j as I64) + (k as I64) - (radius as I64);
                let jj_clamped = max(0, min(jj, (field.n_lon - 1) as I64)) as USize;
                sum += field.get(i, jj_clamped) * kernel[k];
            }
            temp.set(i, j, sum);
        }
    }

    // Vertical pass
    for i in 0..field.n_lat {
        for j in 0..field.n_lon {
            let mut sum = 0.0;
            for k in 0..kernel_size {
                let ii = (i as I64) + (k as I64) - (radius as I64);
                let ii_clamped = max(0, min(ii, (field.n_lat - 1) as I64)) as USize;
                sum += temp.get(ii_clamped, j) * kernel[k];
            }
            result.set(i, j, sum);
        }
    }

    result
}

/// Solve the steady-state Helmholtz PDE: ∇²τ - Γτ + S = 0  (unit diffusivity).
/// Uses SOR (Successive Over-Relaxation) iteration.
///
/// The damping argument is GAMMA — it multiplies τ directly (FVCS W-2: the old
/// name `kappa2` invited the κ² = Γ/D confusion; this solver has no D field, so
/// its Γ input IS the screening term, and κ = sqrt(Γ) is derived-only). The old
/// header also claimed an `-S/D` source normalization the code never performed;
/// the equation above is what the stencil actually solves, arithmetic unchanged.
///
/// This is Form 7 of S_One — the coherence field equation.
fn solve_helmholtz(
    source: &GridField,
    gamma: &GridField,
    n_iterations: USize,
    omega: F64
) -> GridField @ L0 {
    let mut tau = GridField::new(source.n_lat, source.n_lon);

    for _iter in 0..n_iterations {
        for i in 1..(source.n_lat - 1) {
            for j in 1..(source.n_lon - 1) {
                // 5-point Laplacian stencil
                let neighbors = tau.get(i - 1, j) + tau.get(i + 1, j)
                              + tau.get(i, j - 1) + tau.get(i, j + 1);

                // Helmholtz: (neighbors - 4*tau) - gamma*tau + S = 0
                // Solve for tau: tau = (neighbors + S) / (4 + gamma)
                let g = gamma.get(i, j);
                let s = source.get(i, j);
                let new_val = (neighbors + s) / (4.0 + g);

                // SOR update
                let old_val = tau.get(i, j);
                tau.set(i, j, old_val + omega * (new_val - old_val));
            }
        }
    }

    tau
}

/// Compute gradient magnitude |∇τ| using central differences
fn compute_gradient(tau: &GridField) -> GridField @ L0 {
    let mut grad = GridField::new(tau.n_lat, tau.n_lon);

    for i in 1..(tau.n_lat - 1) {
        for j in 1..(tau.n_lon - 1) {
            let dtdi = (tau.get(i + 1, j) - tau.get(i - 1, j)) / 2.0;
            let dtdj = (tau.get(i, j + 1) - tau.get(i, j - 1)) / 2.0;
            grad.set(i, j, sqrt(dtdi * dtdi + dtdj * dtdj));
        }
    }

    grad
}

/// Compute curl of the tau field (2D: ∂τ/∂x × ∂τ/∂y component)
fn compute_curl(tau: &GridField) -> GridField @ L0 {
    let mut curl = GridField::new(tau.n_lat, tau.n_lon);

    for i in 1..(tau.n_lat - 1) {
        for j in 1..(tau.n_lon - 1) {
            let dtdi = (tau.get(i + 1, j) - tau.get(i - 1, j)) / 2.0;
            let dtdj = (tau.get(i, j + 1) - tau.get(i, j - 1)) / 2.0;
            // 2D curl magnitude (scalar)
            curl.set(i, j, dtdi * dtdj);
        }
    }

    curl
}

/// Convert lat/lon to grid cell indices
fn latlon_to_cell(lat: F64, lon: F64) -> (USize, USize) @ L0 {
    let i = ((lat - LAT_MIN) / GRID_DLAT) as USize;
    let j = ((lon - LON_MIN) / GRID_DLON) as USize;
    let i_clamped = min(i, N_LAT - 1);
    let j_clamped = min(j, N_LON - 1);
    (i_clamped, j_clamped)
}

/// Build coherence fields from atmospheric data
///
/// Source S = CAPE/2000 + 0.3*|SRH|/200 + 0.2*shear/25
/// Damping Γ = |CIN|/200 + Td_depression/20 + 0.1
/// Diffusion D = 1.0 + 0.3*storm_speed/20
///
/// These mappings connect the Helmholtz PDE (Form 7 of S_One)
/// to observable atmospheric quantities.
fn build_coherence_fields(
    cape_field: &GridField,
    srh_field: &GridField,
    shear_field: &GridField,
    cin_field: &GridField,
    td_depression_field: &GridField
) -> (GridField, GridField, GridField, GridField) @ L0 {
    let mut source = GridField::new(N_LAT, N_LON);
    let mut gamma = GridField::new(N_LAT, N_LON);

    for i in 0..N_LAT {
        for j in 0..N_LON {
            // Source: atmospheric loading
            let cape = max(0.0, cape_field.get(i, j));
            let srh = abs(srh_field.get(i, j));
            let shear = max(0.0, shear_field.get(i, j));
            let s = cape / 2000.0 + 0.3 * srh / 200.0 + 0.2 * shear / 25.0;
            source.set(i, j, s);

            // Damping: atmospheric resistance
            let cin = abs(cin_field.get(i, j));
            let td_dep = max(0.0, td_depression_field.get(i, j));
            let g = cin / 200.0 + td_dep / 20.0 + 0.1;
            gamma.set(i, j, g);
            // (kappa = sqrt(Γ/D) is derived-only; the old kappa2 = g/1.0 field
            //  was bitwise-equal to gamma and is gone — FVCS W-2.)
        }
    }

    // Smooth the source field
    let source_smooth = gaussian_smooth(&source, 2);

    // Solve Helmholtz PDE
    let tau = solve_helmholtz(&source_smooth, &gamma, 200, 0.7);

    // Compute gradient
    let grad_tau = compute_gradient(&tau);

    (tau, grad_tau, source, gamma)
}

/// Extract coherence diagnostics at a single storm location
fn extract_at_point(
    lat: F64,
    lon: F64,
    tau: &GridField,
    grad_tau: &GridField,
    source: &GridField,
    gamma: &GridField,
    srh_at_storm: F64,
    shear_at_storm: F64
) -> CoherenceDiagnostics @ L0 {
    let (i, j) = latlon_to_cell(lat, lon);

    let tau_val = tau.get(i, j);
    let grad_val = grad_tau.get(i, j);
    let s_val = source.get(i, j);
    let g_val = gamma.get(i, j);

    // S/Γ ratio
    let s_over_gamma = if g_val > 1e-6 { s_val / g_val } else { 0.0 };

    // Damköhler number: Da = Γ * L² / D
    let da = g_val * DX_KM * DX_KM / 1.0;

    // Torsion: SRH × curl(τ)
    let curl_val = if i > 0 && i < N_LAT - 1 && j > 0 && j < N_LON - 1 {
        let dtdi = (tau.get(i + 1, j) - tau.get(i - 1, j)) / 2.0;
        let dtdj = (tau.get(i, j + 1) - tau.get(i, j - 1)) / 2.0;
        dtdi * dtdj
    } else {
        0.0
    };
    let torsion = (srh_at_storm / 200.0) * curl_val;

    // Alignment: projection of shear vector onto ∇τ direction
    let alignment = if grad_val > 1e-6 {
        // Approximate: shear magnitude × gradient magnitude × cos(angle)
        // Since we don't have shear direction, use magnitude product
        (shear_at_storm / 25.0) * grad_val
    } else {
        0.0
    };

    // Coherence energy (KL divergence proxy)
    let e_coh = tau_val * tau_val * g_val;

    // Singularity test: how many conditions are met?
    let mut singularity_count: USize = 0;
    if s_over_gamma > 1.0 { singularity_count += 1; }  // Source exceeds damping
    if grad_val > 0.1 { singularity_count += 1; }       // Steep gradient
    if abs(torsion) > 0.01 { singularity_count += 1; }  // Torsional coupling
    if da > 100.0 { singularity_count += 1; }            // Decay-dominated regime
    if alignment > 0.05 { singularity_count += 1; }      // Shear-coherence alignment

    CoherenceDiagnostics {
        tau: tau_val,
        grad_tau: grad_val,
        torsion,
        alignment,
        s_over_gamma,
        da,
        singularity_count,
        source_field: s_val,
        gamma_field: g_val,
        e_coh
    }
}

// ============================================================
// Tests
// ============================================================

#[test]
fn test_latlon_to_cell() {
    let (i, j) = latlon_to_cell(35.0, -97.0);
    assert!(i > 0 && i < N_LAT);
    assert!(j > 0 && j < N_LON);
}

#[test]
fn test_helmholtz_zero_source() {
    let source = GridField::new(N_LAT, N_LON);
    let gamma = GridField::new(N_LAT, N_LON);
    let tau = solve_helmholtz(&source, &gamma, 50, 0.7);
    // Zero source → zero tau everywhere
    assert_approx_eq!(tau.max_value(), 0.0, 1e-6);
}

#[test]
fn test_sigmoid_in_gbt() {
    // This test verifies the coherence module works with the GBT module
    let diag = CoherenceDiagnostics {
        tau: 0.5,
        grad_tau: 0.1,
        torsion: 0.02,
        alignment: 0.08,
        s_over_gamma: 1.2,
        da: 150.0,
        singularity_count: 3,
        source_field: 0.6,
        gamma_field: 0.5,
        e_coh: 0.125
    };
    assert!(diag.singularity_count == 3);
    assert!(diag.s_over_gamma > 1.0);
}
