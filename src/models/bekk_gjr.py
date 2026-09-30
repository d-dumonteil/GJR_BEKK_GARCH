import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import math as mth

from itertools import combinations
from dataclasses import dataclass

from scipy.stats import jarque_bera, probplot
from scipy.optimize import minimize
from numba import njit

import matplotlib.pyplot as plt

@njit
def multivariate_normal_rvs(mean, cov):
    """Sample from multivariate normal distribution."""
    dim = mean.shape[0]
    z = np.random.randn(dim)
    L = np.linalg.cholesky(cov.T)
    sample = mean + L @ z
    
    return sample

@njit
def forecast_bekk_gjr(y, T_forecast, C, A, B, G, mu):
    """
    Forecast future conditional covariances and residuals for the BEKK-GJR(1,1) model,
    introducing randomness into the covariances by simulating future residuals.

    Args:
        y (np.ndarray): Observed time series (n_obs, n_dims).
        model (MGARCH_GJR): Fitted BEKK-GJR model instance.
        T_forecast (int): Number of forecast periods.

    Returns:
        np.ndarray: Forecasted covariance matrices (T_forecast, n_dims, n_dims).
        np.ndarray: Simulated residuals (T_forecast, n_dims).
    """
    n_obs, n_dims = y.shape
    # Start from the last observed H
    epsilon_past = y[-1] - mu  # last residual

    H_past = compute_bekk_gjr_covariances(y, C, A, B, G, mu)[-1]  # last covariance
    H_forecast = np.zeros((T_forecast, n_dims, n_dims))
    eps_forecast = np.zeros((T_forecast, n_dims))

    H_t = H_past.copy()
    epsilon_t = epsilon_past.copy()

    for t in range(T_forecast):
        # Simulate next epsilon from N(0, H_t)
        epsilon_t = multivariate_normal_rvs(np.zeros(n_dims), H_t)

        # Indicator for negative residuals
        I_neg = np.diag((epsilon_t < 0))

        # Update next covariance
        H_t = (C @ C.T +
               A.T @ np.outer(epsilon_t, epsilon_t) @ A +
               B.T @ H_t @ B +
               G.T @ (np.outer(epsilon_t, epsilon_t) * I_neg) @ G)

        # Regularize for numerical stability
        H_t += 1e-6 * np.eye(n_dims)

        H_forecast[t] = H_t
        eps_forecast[t] = epsilon_t

    return H_forecast, eps_forecast

def compute_standardized_residuals(y, mu, H):
    """
    Compute standardized residuals for the BEKK-GJR model.

    Args:
        y (np.ndarray): Time series (n_obs, n_dims).
        mu (np.ndarray): Mean vector (n_dims,).
        H (np.ndarray): Conditional covariance matrices (n_obs, n_dims, n_dims).

    Returns:
        np.ndarray: Standardized residuals (n_obs, n_dims).
    """
    n_obs, n_dims = y.shape
    residuals = y - mu  # Compute residuals
    std_residuals = np.zeros_like(residuals)

    for t in range(n_obs):
        std_devs = np.sqrt(np.diag(H[t]))  # Extract conditional standard deviations
        std_residuals[t] = residuals[t] / std_devs

    return std_residuals

def extract_variances_and_correlations(H):
    """
    Extract variances and correlations from covariance matrices.

    Args:
        H (np.ndarray): Covariance matrices (n_obs, n_dims, n_dims).

    Returns:
        variances (np.ndarray): Variances (n_obs, n_dims).
        correlations (np.ndarray): Correlations (n_obs, n_dims, n_dims).
    """
    n_obs, n_dims, _ = H.shape
    variances = np.zeros((n_obs, n_dims))
    correlations = np.zeros((n_obs, n_dims, n_dims))

    for t in range(n_obs):
        variances[t] = np.diag(H[t])  # Extract variances
        stddevs = np.sqrt(np.diag(H[t]))
        correlations[t] = H[t] / np.outer(stddevs, stddevs)  # Convert covariance to correlation
        np.fill_diagonal(correlations[t], 1.0)  # Ensure diagonal is exactly 1

    return variances, correlations

@njit
def compute_bekk_gjr_covariances(y, C, A, B, G, mu=None):
    """
    Compute conditional covariance matrices for a BEKK-GJR(1,1) model.

    Args:
        y (np.ndarray): Time series (n_obs, n_dims).
        C, A, B, G (np.ndarray): Model parameters.
        mu (np.ndarray): Mean vector.

    Returns:
        np.ndarray: Conditional covariance matrices (n_obs, n_dims, n_dims).
    """
    n_obs, n_dims = y.shape
    H = np.zeros((n_obs, n_dims, n_dims))  # To store the conditional covariance matrices

    if mu is None:
        mu = np.zeros(n_dims)

    # Initialize
    H[0] = C @ C.T

    for t in range(1, n_obs):
        epsilon = y[t-1] - mu  # Lagged residuals
        I_neg = np.diag((epsilon < 0))  # Indicator for negative residuals

        # Compute the conditional covariance matrix
        H[t] = (C @ C.T +
                A.T @ np.outer(epsilon, epsilon) @ A +
                B.T @ H[t-1] @ B +
                G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        

    return H

@njit
def bekk_gjr_log_likelihood(params, y, n_dims):
    """
    Compute the log-likelihood for the BEKK-GJR(1,1) model.

    Args:
        params (np.ndarray): Flattened model parameters.
        y (np.ndarray): Time series (n_obs, n_dims).
        n_dims (int): Number of dimensions.

    Returns:
        float: Negative log-likelihood.
    """
    n_obs = y.shape[0]

    # Reshape parameters
    C = params[:n_dims**2].reshape(n_dims, n_dims)
    A = params[n_dims**2:n_dims**2*2].reshape(n_dims, n_dims)
    B = params[n_dims**2*2:n_dims**2*3].reshape(n_dims, n_dims)
    G = params[n_dims**2*3:].reshape(n_dims, n_dims)

    # Compute conditional covariances
    H = compute_bekk_gjr_covariances(y, C, A, B, G)

    # Compute log-likelihood
    ll = 0
    for t in range(n_obs):
        epsilon_t = y[t]
        ll += (np.log(np.linalg.det(H[t])) +
                epsilon_t.T @ np.linalg.inv(H[t]) @ epsilon_t)

    return 0.5 * ll  # Return negative log-likelihood for minimization




@dataclass
class MGARCH_GJR:
    C: np.ndarray
    A: np.ndarray
    B: np.ndarray
    G: np.ndarray
    mu: np.ndarray

    @classmethod
    def from_params(cls, params, n_dims):
        C = params[:n_dims**2].reshape(n_dims, n_dims)
        A = params[n_dims**2:n_dims**2*2].reshape(n_dims, n_dims)
        B = params[n_dims**2*2:n_dims**2*3].reshape(n_dims, n_dims)
        G = params[n_dims**2*3:].reshape(n_dims, n_dims)
        return cls(C, A, B, G, np.zeros(n_dims))

    def conditional_covariances(self, y):
        """
        Compute conditional covariances for the fitted model.

        Args:
            y (np.ndarray): Time series (n_obs, n_dims).

        Returns:
            np.ndarray: Conditional covariance matrices (n_obs, n_dims, n_dims).
        """
        return compute_bekk_gjr_covariances(y, self.C, self.A, self.B, self.G, self.mu)

    def standardized_residuals(self, y):
        """
        Compute standardized residuals for the time series.

        Args:
            y (np.ndarray): Observed time series (n_obs, n_dims).

        Returns:
            np.ndarray: Standardized residuals (n_obs, n_dims).
        """
        H = self.conditional_covariances(y)
        return compute_standardized_residuals(y, self.mu, H)

    def log_likelihood(self, y):
        """
        Compute the log-likelihood for the model and the time series.

        Args:
            y (np.ndarray): Time series (n_obs, n_dims).

        Returns:
            float: Log-likelihood value.
        """
        H = self.conditional_covariances(y)
        ll = 0

        for t in range(len(y)):
            epsilon = y[t] - self.mu
            ll += (np.log(np.linalg.det(H[t])) +
                   epsilon.T @ np.linalg.inv(H[t]) @ epsilon)

        return -0.5 * ll
    
    def sample_forecast(self, y, T_forecast=30, n_samples=100):
        """
        Generate Monte Carlo sample forecasts with randomness in both residuals and covariances.
        At each step, new residuals are drawn from N(0, H_t), influencing future covariances.

        Args:
            y (np.ndarray): Observed time series (n_obs, n_dims).
            T_forecast (int): Number of forecast periods.
            n_samples (int): Number of Monte Carlo samples.

        Returns:
            np.ndarray: Sample forecasted time series (n_samples, T_forecast, n_dims).
            np.ndarray: Corresponding forecasted covariances (n_samples, T_forecast, n_dims, n_dims).
        """
        n_dims = self.C.shape[0]
        samples = np.zeros((n_samples, T_forecast, n_dims))
        H_samples = np.zeros((n_samples, T_forecast, n_dims, n_dims))

        for n in range(n_samples):
            H_forecast, eps_forecast = forecast_bekk_gjr(y, T_forecast, self.C, self.A, self.B, self.G, self.mu)
            H_samples[n] = H_forecast
            # Given eps_forecast are the simulated residuals, we simulate y from mu + eps_forecast
            # Actually, eps_forecast are already from N(0,H_t). To get y, add mu:
            # Note: In a GARCH model, epsilon_t = y_t - mu. We drew epsilon_t ~ N(0,H_t)
            # thus y_t = mu + epsilon_t.
            samples[n] = self.mu + eps_forecast

        return samples, H_samples
    
def extract_correlation_matrix(H):
    """
    Extract correlation matrices from covariance matrices.

    Args:
        H (np.ndarray): Covariance matrices (n_obs, n_dims, n_dims).

    Returns:
        np.ndarray: Correlation matrices (n_obs, n_dims, n_dims).
    """
    n_obs, n_dims, _ = H.shape
    correlations = np.zeros((n_obs, n_dims, n_dims))

    for t in range(n_obs):
        stddevs = np.sqrt(np.diag(H[t]))
        correlations[t] = H[t] / np.outer(stddevs, stddevs)
        np.fill_diagonal(correlations[t], 1.0)  # Ensure diagonal is exactly 1

    return correlations

def fit_bekk_gjr(y, n_dims):
    """
    Fit the BEKK-GJR(1,1) model to data.

    Args:
        y (np.ndarray): Time series (n_obs, n_dims).
        n_dims (int): Number of dimensions.

    Returns:
        scipy.optimize.OptimizeResult: Optimization result containing estimated parameters.
    """
    # Initial parameter guesses
    C_init = np.eye(n_dims) * 0.1  # Lower triangular constant matrix
    A_init = np.eye(n_dims) * 0.1  # Coefficient for residuals
    B_init = np.eye(n_dims) * 0.1  # Coefficient for covariances
    G_init = np.eye(n_dims) * 0.1  # Coefficient for leverage effects
    initial_params = np.concatenate([C_init.flatten(), A_init.flatten(), B_init.flatten(), G_init.flatten()])

    def objective(params, y, n_dims):
        """
        Objective function for the BEKK-GJR model.

        Args:
            params (np.ndarray): Flattened model parameters.
            y (np.ndarray): Time series (n_obs, n_dims).
            n_dims (int): Number of dimensions.

        Returns:
            float: Negative log-likelihood.
        """
        try:
            return bekk_gjr_log_likelihood(params, y, n_dims)
        except np.linalg.LinAlgError:
            return np.inf
    
    # Minimize the negative log-likelihood
    result = minimize(
        objective,  # Objective function
        initial_params,           # Initial guess
        args=(y, n_dims),         # Additional arguments to the function
        options={"disp": True, "maxiter": 500}  # Display progress and limit iterations
    )

    return result

def normality_tests(std_residuals, syms):
    """
    Perform normality tests on standardized residuals.

    Args:
        std_residuals (np.ndarray): Standardized residuals (n_obs, n_dims).
    """
    n_dims = std_residuals.shape[1]

    for i in range(n_dims):
        # Jarque-Bera Test
        jb_stat, jb_pval = jarque_bera(std_residuals[:, i])
        print(f"Dim {i+1} - Jarque-Bera Test: stat={jb_stat:.2f}, p-value={jb_pval:.4f}")

        # QQ Plot
        probplot(std_residuals[:, i], dist="norm", plot=plt)
        plt.title(f"QQ Plot ({syms[i]})")
        plt.grid()
        plt.show()

def plot_autocorrelation(std_residuals, syms):
    """
    Plot autocorrelation and partial autocorrelation for each dimension.

    Args:
        std_residuals (np.ndarray): Standardized residuals (n_obs, n_dims).
    """
    n_dims = std_residuals.shape[1]

    for i in range(n_dims):
        sm.graphics.tsa.plot_acf(std_residuals[:, i], lags=20, title=f"ACF ({syms[i]})")
        sm.graphics.tsa.plot_pacf(std_residuals[:, i], lags=20, title=f"PACF ({syms[i]})")
        plt.show()
        
def plot_standardized_residuals(std_residuals, syms):
    """
    Plot standardized residuals for each dimension.

    Args:
        std_residuals (np.ndarray): Standardized residuals (n_obs, n_dims).
    """
    n_obs, n_dims = std_residuals.shape
    plt.figure(figsize=(12, 6))

    for i in range(n_dims):
        plt.plot(std_residuals[:, i], label=f"Standardized Residual ({syms[i]})")
    
    plt.title("Standardized Residuals")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # 8 stocks CAC40
    # Air Liquide, AXA, BNP Paribas, Carrefour, Crédit Agricole, LVMH, Safran, TotalEnergies
    syms = sorted(['AI.PA', 'CS.PA'])#'ACA.PA', 'MC.PA', 'SAF.PA', 'TTE.PA'])
    n_dims = len(syms)
    fit = True
    test_size = 30
    p = 1
    q = 1

    data = yf.download(syms, start='2010-01-01', end='2025-12-06',auto_adjust=True)["Close"]
    y = np.log(data).diff().dropna()

    y_matrix = y.values
    timestamps = y.index
    test_size = mth.floor(0.15*len(y_matrix))

    y_train = y_matrix[:-test_size]

    result = fit_bekk_gjr(y_train, n_dims)
    model = MGARCH_GJR.from_params(result.x, n_dims)
    
    np.savez_compressed("cov_and_modelparams.npz", n_dims=n_dims,H_train=compute_bekk_gjr_covariances(y_train, model.C, model.A, model.B, model.G), y=y_matrix,test_size=test_size, C=model.C, A=model.A, B=model.B, G=model.G)
    

if __name__ == "__main__":
    main()
