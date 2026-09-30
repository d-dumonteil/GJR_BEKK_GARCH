# 📈 Quantitative Portfolio Optimization: BEKK-GJR(1,1) MGARCH + VAR

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://projetetude28.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready quantitative research and portfolio construction pipeline combining **Multivariate Asymmetric GARCH (BEKK-GJR)** for dynamic conditional covariance forecasting, **Vector Autoregression (VAR)** for expected returns, and **Convex Optimization with L1 Turnover Penalization (CVXPY)** under realistic market friction constraints.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture & Modular Structure](#architecture--modular-structure)
- [Mathematical & Theoretical Foundations](#mathematical--theoretical-foundations)
  - [Volatility & Covariance Engine: BEKK-GJR(1,1)](#volatility--covariance-engine-bekk-gjr11)
  - [Alpha Generation: Vector Autoregression (VAR)](#alpha-generation-vector-autoregression-var)
  - [Portfolio Optimization with Turnover Penalty](#portfolio-optimization-with-turnover-penalty)
  - [Execution & Friction Dynamics](#execution--friction-dynamics)
- [Backtest Strategies](#backtest-strategies)
- [Installation & Quickstart](#installation--quickstart)
- [Interactive Dashboard](#interactive-dashboard)
- [Testing & Quality Assurance](#testing--quality-assurance)

---

## Overview

Traditional mean-variance frameworks suffer from parameter instability, static correlation assumptions, and severe sensitivity to estimation risk. This repository implements an end-to-end quantitative framework designed to solve these caveats:

1. **Dynamic Risk Modeling**: Covariances are updated conditionally using a full BEKK-GJR model capturing volatility clustering, time-varying correlation, and the leverage effect (asymmetric market reaction to negative innovations).
2. **High-Performance Computation**: Quasi-Maximum Likelihood Estimation (QMLE) accelerated through **Numba JIT compilation**, reducing computation time on high-dimensional assets.
3. **Friction-Aware Optimization**: Mean-variance utility maximized via quadratic programming (**OSQP solver via CVXPY**), featuring L1-norm turnover shrinkage to mitigate transaction drag.
4. **Execution Modeling**: Realistic simulation comparing unconstrained, gross, and turnover-regularized strategies against a benchmark ($1/N$).

---

## Architecture & Modular Structure

```text
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py          # Yahoo Finance ETL pipeline & local caching
│   │   └── tickers.py          # Asset universes (CAC 40, US Large Caps, ETFs, Crypto)
│   ├── models/
│   │   ├── __init__.py
│   │   └── bekk_gjr.py         # Numba JIT-compiled BEKK-GJR estimation & simulation
│   └── backtest/
│       ├── __init__.py
│       ├── engine.py           # CVXPY optimization & backtest simulation loops
│       └── metrics.py          # Realized Sharpe, Drawdown, and volatility analytics
├── app/
│   └── dashboard.py            # Streamlit web application
├── tests/
│   ├── __init__.py
│   └── test_portfolio.py       # Unit tests (weight bounds, budget constraints)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Mathematical & Theoretical Foundations

### Volatility & Covariance Engine: BEKK-GJR(1,1)

The conditional covariance matrix $H_t$ is parameterized in Baba-Engle-Kraft-Kroner (BEKK) form to guarantee positive definiteness without imposing restrictive diagonal-only assumptions:

$$
H_t = CC^\top + A^\top \varepsilon_{t-1}\varepsilon_{t-1}^\top A + B^\top H_{t-1} B + G^\top \left(\varepsilon_{t-1}\varepsilon_{t-1}^\top \odot I_{t-1}^-\right) G
$$

Where:

- $C \in \mathbb{R}^{K \times K}$ is a lower triangular constant matrix ($CC^\top$ is the baseline unconditional covariance).
- $A \in \mathbb{R}^{K \times K}$ governs the ARCH effect (responsiveness to past innovations).
- $B \in \mathbb{R}^{K \times K}$ captures the GARCH effect (volatility persistence).
- $G \in \mathbb{R}^{K \times K}$ models the asymmetric leverage effect, where $I_{t-1}^- = \text{diag}(\mathbf{1}_{\{\varepsilon_{t-1} < 0\}})$.
- $\odot$ denotes the element-wise Hadamard product.

Parameter estimation is performed by minimizing the negative Gaussian log-likelihood:

$$
\mathcal{L}(\theta) = \frac{1}{2} \sum_{t=1}^T \left( \ln \vert H_t \vert + \varepsilon_t^\top H_t^{-1} \varepsilon_t \right)
$$

### Alpha Generation: Vector Autoregression (VAR)

Expected short-term drift $\hat{\mu}_{t+1}$ is projected using a multivariate vector autoregressive process:

$$
y_t = \nu + \sum_{l=1}^p \Phi_l y_{t-l} + u_t
$$

At each rebalancing date $t$, the VAR process is fit strictly over past observations to eliminate look-ahead bias, generating 1-step-ahead forecasted returns:

$$
\hat{\mu}_{t+1} = \mathbb{E}[y_{t+1} \mid \mathcal{F}_t]
$$

### Portfolio Optimization with Turnover Penalty

Given the forecasted covariance $H_{t+1}$ and expected return vector $\hat{\mu}_{t+1}$, optimal weights $w_t \in \mathbb{R}^K$ are determined by solving a convex quadratic optimization problem with an L1 transaction cost penalty:

$$
\max_{w_t} \quad \hat{\mu}_{t+1}^\top w_t - \frac{\gamma}{2} w_t^\top \left(H_{t+1} + \delta I\right) w_t - \lambda_{tc} \Vert w_t - w_{t-1} \Vert_1
$$

$$
\text{subject to} \quad \sum_{i=1}^K w_{i,t} = 1, \quad w_{\min} \le w_{i,t} \le w_{\max}
$$

- $\delta I$ ($\delta = 10^{-6}$) acts as a Tikhonov regularizer ensuring numerical conditioning.
- $\lambda_{tc}$ denotes the marginal cost of transaction per unit of portfolio turnover.

### Execution & Friction Dynamics

Portfolio values evolve using exact arithmetic returns derived from observed log-returns $y_{i,t}$:

$$
R_{i,t} = \exp(y_{i,t}) - 1
$$

$$
V_t = \left(V_{t-1} - \text{Fee}_t\right) \left(1 + w_t^\top R_t\right)
$$

Where transaction costs scale with aggregate rebalancing turnover:

$$
\text{Fee}_t = V_{t-1} \cdot \lambda_{tc} \sum_{i=1}^K \vert w_{i,t} - w_{i,t-1} \vert
$$

---

## Backtest Strategies

| Strategy | Logic | Rebalancing Protocol | Friction Handling |
|---|---|---|---|
| **All-In** | Lump-sum capital allocation at $t_0$. | Daily full rebalance via predicted $(H_{t+1}, \hat{\mu}_{t+1})$. | Explicit turnover tracking (0.5% fee baseline). |
| **REGU (DCA)** | Systematic recurring cash injection. | Full portfolio rebalancing on existing + new equity. | Scaled proportional turnover cost (0.1% fee). |
| **OnlyREGU** | Periodic cash injection without global turnover. | Only incremental capital is steered to optimal assets. | Drastically minimal fee impact; legacy capital drifts. |
| **$1/N$ Benchmark** | Equal-weighted naive baseline. | Buy and hold / equal allocation. | Zero rebalancing friction. |

---

## Installation & Quickstart

### Prerequisites

- Python 3.9, 3.10, or 3.11
- GCC/Clang (for Numba JIT compilation)

### Setup

```bash
# Clone the repository
git clone https://github.com/votre-utilisateur/votre-repo.git
cd votre-repo

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Pipeline via CLI

Fetch historical data:

```bash
python -m src.data.loaders
```

Estimate the BEKK-GJR Model:

```bash
python -m src.models.bekk_gjr
```

Execute Backtesting Engine:

```bash
python -m src.backtest.engine
```

---

## Interactive Dashboard

An interactive dashboard built with Streamlit provides portfolio analytics, parameter controls, asset selection, and comparative risk metrics.

Run locally:

```bash
streamlit run app/dashboard.py
```

Or access the live application directly: [projetetude28.streamlit.app](https://projetetude28.streamlit.app/)

---

## Testing & Quality Assurance

Unit tests ensure mathematical integrity (portfolio budget constraints, weight boundaries) and robust code quality:

```bash
# Run test suite
pytest -v tests/
```

Expected output:

```text
tests/test_portfolio.py::test_weights_sum_to_one PASSED          [ 50%]
tests/test_portfolio.py::test_weights_bounds PASSED              [100%]
============================== 2 passed in 0.42s ===============================
```
