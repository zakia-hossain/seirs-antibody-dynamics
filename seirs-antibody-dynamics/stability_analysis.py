# ----------------------------------------------------------------------------------------------------------------
# This script calculate 2d plot and periond using FFT for unstable and eigen process for stable regions with FFT
# ----------------------------------------------------------------------------------------------------------------
# pertussis parameter, Measeals, Influenza

import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.stats import norm
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpmath import mp
from scipy.signal import find_peaks

mp.dps = 50

# ------------------------------------------------------------------
# Plot style
# ------------------------------------------------------------------
fontsize = 12
params = {
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.labelsize': fontsize,
    'font.size': fontsize,
    'axes.titlesize': fontsize,
    'legend.fontsize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'text.usetex': True,
}
plt.rcParams.update(params)

# # ---------------------------
# # Fixed parameters (pertussis)
# # ---------------------------
mu    = 1/70
sigma = 365/8
gamma = 365/15
beta  = 1.1344 * 365
eps   = 1e-16
N     = 1000000
l     = 0.0
x_a     = 1.0
delta = 0.2  # for p(alpha)



# ---------------------------
# Functions
# ---------------------------
def p_of_alpha(alpha, nu):
    return norm.cdf(np.log(nu), loc=np.log(alpha), scale=delta)

def pprime_of_alpha(alpha, nu):
    pdf_val = norm.pdf(np.log(nu), loc=np.log(alpha), scale=delta)
    return - pdf_val / alpha

def S_of_I(I):
    return mu * N / (mu + (beta * I) / N)

def E_of_I(I):
    return (gamma + mu) * I / sigma

def R_of_I_alpha(I, alpha, nu):
    p = p_of_alpha(alpha, nu)
    return gamma * I / (mu + (beta * I / N) * p)

def dS_dI(I):
    den = mu + (beta * I) / N
    return - mu * beta / (den * den)

def dR_dI(I, alpha, nu):
    p = p_of_alpha(alpha, nu)
    den = mu + (beta * I / N) * p
    return gamma * mu / (den * den)

def dR_dalpha(I, alpha, nu):
    p  = p_of_alpha(alpha, nu)
    pp = pprime_of_alpha(alpha, nu)
    den = mu + (beta * I / N) * p
    return - gamma * I * (beta * I / N) * pp / (den * den)

# Residuals (λ=0 here)
def residuals(z, nu, c):
    I, alpha = z
    S = S_of_I(I)
    R = R_of_I_alpha(I, alpha, nu)
    p = p_of_alpha(alpha, nu)
    F1 = beta / N * (S + p * R) - (sigma + mu) * (gamma + mu) / sigma
    q  = (gamma * I) / (R + eps)
    F2 = -c * (alpha - l) + q * (x_a - alpha)
    return np.array([F1, F2])

def jacobian(z, nu, c):
    I, alpha = z
    R = R_of_I_alpha(I, alpha, nu)
    p = p_of_alpha(alpha, nu)
    pp  = pprime_of_alpha(alpha, nu)
    dS  = dS_dI(I)
    dRI = dR_dI(I, alpha, nu)
    dRa = dR_dalpha(I, alpha, nu)

    F1_I = beta / N * (dS + p * dRI)
    F1_a = beta / N * (pp * R + p * dRa)

    q = (gamma * I) / (R + eps)
    dq_dI = gamma / (R + eps) - (gamma * I) / ((R + eps) ** 2) * dRI
    dq_da = - (gamma * I) / ((R + eps) ** 2) * dRa

    F2_I = dq_dI * (x_a - alpha)
    F2_a = -c + dq_da * (x_a - alpha) - q
    return np.array([[F1_I, F1_a],
                     [F2_I, F2_a]])

def full_jacobian(S, E, I, R, a, nu, c):
    p  = p_of_alpha(a, nu)
    pp = pprime_of_alpha(a, nu)
    bN = beta / N
    J = np.zeros((5,5))
    J[0,0] = -(mu + bN*I)
    J[0,2] = -bN*S
    J[1,0] =  bN*I
    J[1,1] = -(sigma + mu)
    J[1,2] =  bN*(S + R*p)
    J[1,3] =  bN*I*p
    J[1,4] =  bN*I*R*pp
    J[2,1] =  sigma
    J[2,2] = -(gamma + mu)
    J[3,2] =  gamma - bN*R*p
    J[3,3] = -(mu + bN*I*p)
    J[3,4] = - bN*I*R*pp
    q = (gamma * I) / (R + eps)
    J[4,2] =  (gamma / (R + eps)) * (x_a - a)
    J[4,3] = -(gamma * I) / ((R + eps)**2) * (x_a - a)
    J[4,4] = -c - q
    return J

# ODE for time simulation (λ=0)
def seirs_diff(t, pop, m, n, beta0, sigma, gamma, mu, c, thr, sigma_ln):
    S = pop[0]
    E = pop[1:m+1]
    I = pop[m+1:m+n+1]
    R = pop[m+n+1]
    A = pop[m+n+2]
    SumI = np.sum(I)
    Nloc = 1e6
    dPop = np.zeros(m + n + 3)
    def f_n(A, thr, sigma_ln):
        return norm.cdf(np.log(thr), loc=np.log(A), scale=sigma_ln)

    dPop[0] = mu * (Nloc - S) - S * beta0 * SumI / Nloc
    dPop[1] = (S + R * f_n(A, thr, sigma_ln)) * beta0 * SumI / Nloc - (m * sigma + mu) * E[0]
    for i in range(1, m):
        dPop[1 + i] = m * sigma * E[i-1] - (m * sigma + mu) * E[i]
    dPop[m+1] = m * sigma * E[-1] - (n * gamma + mu) * I[0]
    for j in range(1, n):
        dPop[m+1+j] = n * gamma * I[j-1] - (n * gamma + mu) * I[j]
    dPop[m+n+1] = n * gamma * I[-1] - R * f_n(A, thr, sigma_ln) * beta0 * SumI / Nloc - mu * R
    dPop[m+n+2] = -c * A + (gamma * SumI / (R + 1e-16)) * (1 - A) if R > 0 else 0.0
    return dPop

def eigs_and_period(J,
                    m, n, beta0, sigma0, gamma0, mu0, c0, nu0, sigma_ln,
                    S0=200000, E0sum=5000, I0sum=3000, R0=792000, A0=0.5,
                    max_years=200, samples_per_year=365,
                    require_overall_stability=True,
                    tol_real=1e-10, tol_imag=1e-12):
    """
    Returns:
        w, max_real, T (eigs period or np.nan), dom_period (FFT period or np.nan)
    """
    # --- eigen route ---
    w = np.linalg.eigvals(J)
    re = np.real(w)
    im = np.imag(w)
    max_real = float(np.max(re))
    T = np.nan
    dom_period = np.nan

    fully_stable = np.all(re < -tol_real) if require_overall_stability else True
    has_complex  = np.any(np.abs(im) > tol_imag)

    if fully_stable and has_complex:
        omega = float(np.min(np.abs(im[np.abs(im) > tol_imag])))  # rad/year
        T = 2.0 * np.pi / omega

    # --- FFT fallback if not using eigen period ---
    if not (fully_stable and has_complex):

        E0 = np.array([E0sum / m] * m)
        I0 = np.array([I0sum/ n] * n)
       
        Last0 = np.concatenate(([S0], E0, I0, [R0, A0]))

        t_eval = np.linspace(0, max_years, samples_per_year * max_years + 1)
        AbsTol = [1e-6] + [1e-13] * m+ [1e-13] * n + [1e-6, 1e-6]
        sol = solve_ivp(
            seirs_diff, [0, max_years], Last0, t_eval=t_eval,
            args=(m, n, beta0, sigma0, gamma0, mu0, c0, nu0, sigma_ln),
            method='Radau', atol=AbsTol, rtol=1e-5
        )
        pop = sol.y.T
        I_vals = np.sum(pop[:, 1+m:1+m+n], axis=1)
        I_last = I_vals[-24 * 365:]
        rounded = np.round(I_last, 2)
        unique_vals = np.unique(rounded)

        if len(unique_vals) == 1:
            print("Infectious class flat in last 24 years — no oscillation.")
            dom_period[i_idx, j_idx] = 0.0
        else:
            print(f"Detected variation: {len(unique_vals)} unique values (2 decimals)")

            # Detrend
            I_til = I_last - np.mean(I_last)
            U = np.fft.fft(I_til)
            L = len(U)
            S_oneside = U[:L//2] / (L/2)
            freqs = 365 * np.arange(L//2) / L
            S_mag = np.abs(S_oneside)

            dom_index = np.argmax(S_mag[1:]) + 1  # skip zero frequency
            freq = freqs[dom_index]
            fft_period = 1 / freq if freq > 0 else 0.0
            dom_period = fft_period
    return w, max_real, T, dom_period,fully_stable

# ===================== MAIN GRID =====================

durations = np.arange(1, 51, 1)
m = n = 1

initial_guess = [200000, 5000, 3000, 792000, 0.5]
I0, alpha0 = float(initial_guess[2]), float(initial_guess[4])

durations_stable   = []   # where all Re(λ) < 0
most_neg_vals      = []   # max real part (least negative)
tau_e_fold         = []   # -1 / most_neg_vals
T_osc_list         = []   # oscillation period if a complex stable pair exists, else NaN
re_os_list =[]
last_I, last_a = I0, alpha0

for dur in durations:
    c  = 1.0 / dur
    nu=0.15

    # (I,alpha) equilibrium solve
    lb, ub = np.array([1e-12, 1e-12]), np.array([N, 1.0])
    z0 = np.array([last_I, last_a])
    sol = least_squares(
        residuals, z0, jac=jacobian, args=(nu, c),
        bounds=(lb, ub), method="trf",
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000
    )
    if not sol.success:
        # keep alignment with arrays
        continue

    I_star, alpha_star = sol.x
    S_star = S_of_I(I_star)
    E_star = E_of_I(I_star)
    R_star = R_of_I_alpha(I_star, alpha_star, nu)

    # warm start
    last_I, last_a = I_star, alpha_star

    # 5x5 Jacobian and eigen analysis
    J = full_jacobian(S_star, E_star, I_star, R_star, alpha_star, nu, c)
    w = np.linalg.eigvals(J)
    re = np.real(w)
    im = np.imag(w)

    if np.all(re < 0.0):
        # stable: store the least-negative real part and e-fold time
        lam_max = np.max(re)              # closest to 0 from below
        durations_stable.append(dur)
        most_neg_vals.append(lam_max)
        tau_e_fold.append(-1.0 / lam_max)  # years

        # oscillatory info if a complex stable pair exists
        mask_complex = np.abs(im) > 1e-10
        if np.any(mask_complex):
            idx= np.argmin(np.abs(im[mask_complex]))
            complex_indices=np.where(mask_complex)[0]
            chosen_idx= complex_indices[idx]
            lam=w[chosen_idx]
            decay = np.real(lam)
            re_os_list.append(decay)

            omega = np.min(np.abs(im[mask_complex]))    # rad/year
            T_osc = 2.0 * np.pi / omega                 # years
            T_osc_list.append(T_osc)


# --- Plots ---
# # --- Plot ---
plt.figure(figsize=(7,5))
plt.plot(durations_stable , most_neg_vals, 'o-', lw=2)
plt.xlabel("Immune duration (years)")
plt.ylabel("Most negative eigenvalue real part")
plt.show()

plt.figure(figsize=(7,5))
plt.plot(durations_stable , re_os_list, 'o-', lw=2)
plt.xlabel("Immune duration (years)")
plt.ylabel("Negative real part associated with imaginary part of eigenvalue ")
plt.show()
fig, ax1 = plt.subplots(figsize=(8,5))

ax1.plot(durations_stable, most_neg_vals, 'o-', lw=2, label=r'$\max \Re(\lambda)$ (stable)')
ax1.set_xlabel("Immune duration (years)")
ax1.set_ylabel(r"Least-negative real part  $\max \Re(\lambda)$  (1/years)")
ax1.grid(True, ls='--', alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(durations_stable, tau_e_fold, 's--', lw=2, color='tab:orange',
         label=r'e-folding time  $\tau=-1/\max \Re(\lambda)$')
ax2.set_ylabel(r'e-folding time (years)')

# Build a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='best')

plt.tight_layout()
plt.show()


##2) Oscillation period where a complex stable pair exists (optional view)
plt.figure(figsize=(8,4))
plt.plot(durations_stable, T_osc_list, 'o-', lw=2, color='tab:green')
plt.xlabel("Immune duration (years)")
plt.ylabel("Oscillation period from eigenvalues (years)")
plt.tight_layout()
plt.show()
# ================================================================
# Largest real part of eigenvalues:
# fixed nu = 0.15 and immune duration from 20 to 1 years
# ================================================================

nu = 0.15

# Immune durations: 20, 19, ..., 1 years
durations = np.arange(20, 0, -1, dtype=float)

largest_real_parts = []
successful_durations = []

# Initial equilibrium guess: [I, alpha]
last_I = 3000.0
last_alpha = 0.5

for dur in durations:

    # Antibody waning rate
    c = 1.0 / dur

    # Bounds for equilibrium variables (I, alpha)
    lower_bound = np.array([1.0e-12, 1.0e-12])
    upper_bound = np.array([N, 1.0])

    # Warm-start from the previous duration
    initial_equilibrium_guess = np.array(
        [last_I, last_alpha]
    )

    # ------------------------------------------------------------
    # Solve the equilibrium equations
    # ------------------------------------------------------------
    equilibrium_solution = least_squares(
        residuals,
        initial_equilibrium_guess,
        jac=jacobian,
        args=(nu, c),
        bounds=(lower_bound, upper_bound),
        method="trf",
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=2000,
    )

    if not equilibrium_solution.success:
        print(
            f"Equilibrium solve failed for "
            f"duration = {dur:.0f} years"
        )
        continue

    # Equilibrium values
    I_star, alpha_star = equilibrium_solution.x

    S_star = S_of_I(I_star)
    E_star = E_of_I(I_star)
    R_star = R_of_I_alpha(
        I_star,
        alpha_star,
        nu,
    )

    # Use this equilibrium as the next initial guess
    last_I = I_star
    last_alpha = alpha_star

    # ------------------------------------------------------------
    # Compute full 5-by-5 Jacobian and its eigenvalues
    # ------------------------------------------------------------
    J = full_jacobian(
        S_star,
        E_star,
        I_star,
        R_star,
        alpha_star,
        nu,
        c,
    )

    eigenvalues = np.linalg.eigvals(J)

    # Largest real part among all eigenvalues
    largest_real_part = np.max(
        np.real(eigenvalues)
    )

    successful_durations.append(dur)
    largest_real_parts.append(largest_real_part)

    print(
        f"duration = {dur:5.1f}, "
        f"I* = {I_star:12.6e}, "
        f"alpha* = {alpha_star:12.6e}, "
        f"max Re(lambda) = {largest_real_part:12.6e}"
    )


# Convert lists to NumPy arrays
successful_durations = np.asarray(
    successful_durations,
    dtype=float,
)

largest_real_parts = np.asarray(
    largest_real_parts,
    dtype=float,
)


# ----------------------------------------------------------------
# Plot largest real part of eigenvalues
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    successful_durations,
    largest_real_parts,
    color="black",
    markersize=4,
    linewidth=1.7,
)

# Zero stability boundary
ax.axhline(
    y=0.0,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Zero Line",
)

# Put duration 20 on the left and duration 1 on the right
ax.set_xlim(20, 1)

# Display integer duration ticks
ax.set_xticks(
    np.arange(20, 0, -1)
)

ax.set_xlabel(
    r"\textbf{Immune duration (years)}"
)

ax.set_ylabel(
    r"\textbf{Largest real part of eigenvalues}"
)

ax.grid(True)
ax.legend(loc="upper right")

fig.tight_layout()
plt.show()