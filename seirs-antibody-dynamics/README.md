# Stability Analysis of an SEIRS Model with Antibody Dynamics

## Overview

This project investigates the effects of immune duration and antibody waning on the stability of an SEIRS infectious disease model. The model incorporates antibody-dependent reinfection and is used to study equilibrium stability and oscillatory disease dynamics.

## Mathematical and Computational Methods

The analysis uses:

- **Nonlinear equilibrium computation:** SciPy's `least_squares` method.
- **Stability analysis:** Eigenvalues of the analytical Jacobian matrix.
- **Oscillation analysis:** Estimation of oscillation periods from complex eigenvalues, with an additional FFT-based routine.
- **Parameter exploration:** Investigation of how immune duration influences system stability.

The model is implemented in Python using NumPy, SciPy, and Matplotlib.

## Running the Code

Install the required packages:

```bash
pip install numpy scipy matplotlib sympy mpmath
```

Run the Python script:

```bash
python stability_analysis.py
```

The program calculates endemic equilibria, evaluates Jacobian eigenvalues, and generates plots illustrating how stability changes with immune duration.

## Research Context

This computational study is part of ongoing research on mathematical modeling of infectious diseases with immune memory and antibody dynamics.

**Author:** Zakia Hossain  
Ph.D. Candidate in Applied Mathematics  
University of New Mexico
