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
- [Repository Structure](#repository-structure)
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
