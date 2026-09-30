import numpy as np
from src.backtest.engine import optimal_sharpe_weights

def test_weights_sum_to_one():
    # Test unitaire : vérifier que les poids du portefeuille somment bien à 1
    mu = np.array([0.02, 0.03])
    sigma = np.array([[0.04, 0.005], [0.005, 0.03]])
    weights = optimal_sharpe_weights(mu, sigma)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)

def test_weights_bounds():
    # Test unitaire : vérifier que les poids respectent les bornes [-1, 1]
    mu = np.array([0.05, -0.02])
    sigma = np.eye(2) * 0.05
    weights = optimal_sharpe_weights(mu, sigma)
    assert np.all(weights >= -1.0 - 1e-4)
    assert np.all(weights <= 1.0 + 1e-4)