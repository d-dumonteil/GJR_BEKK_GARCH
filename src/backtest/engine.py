import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import cvxpy as cp
import matplotlib.dates as mdates

# --- Configuration Globale ---
RF_ANNUAL_LOG = np.log(1 + 0.03)
N_TRADING_DAYS = 252
RF_DAILY_LOG = RF_ANNUAL_LOG / N_TRADING_DAYS


def realized_sharpe_from_portfolio_values(portfolio_values, rf_log=RF_DAILY_LOG):
    portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) < 2:
        return 0.0
    returns = portfolio_values[1:] / portfolio_values[:-1] - 1
    excess = returns - (np.exp(rf_log) - 1)
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return np.sqrt(N_TRADING_DAYS) * np.mean(excess) / std


def predict_next_returns_from_returns(returns, lags=3):
    returns = np.array(returns)
    if len(returns) <= lags + 1:
        return np.zeros(returns.shape[1])
    try:
        model = VAR(returns)
        model_fit = model.fit(lags)
        forecast = model_fit.forecast(returns[-lags:], steps=1)
        return forecast[0]
    except Exception:
        return np.mean(returns[-min(15, len(returns)):], axis=0)


def optimal_sharpe_weights(mu, Sigma, prev_weights=None, lambda_tc=0.0):
    n = len(mu)
    w = cp.Variable(n)

    Sigma_reg = Sigma + 1e-6 * np.eye(n)
    risk = cp.quad_form(w, Sigma_reg)

    turnover_penalty = 0
    if prev_weights is not None and lambda_tc > 0:
        turnover_penalty = lambda_tc * cp.norm1(w - prev_weights)

    objective = cp.Maximize(mu @ w - 0.5 * risk - turnover_penalty)
    constraints = [cp.sum(w) == 1, w >= -1, w <= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if w.value is None:
        return np.ones(n) / n
    return np.array(w.value).flatten()


# --- Stratégie 1 : ALL-IN ---
def strat_all_in(n_dims, test_size, y, H_train, A, B, C, G, initial_cash):
    """
    Retourne 4 tableaux : (sans_frais, brut, opti, ref)
    """
    # BUG FIX : copie locale pour ne pas muter la liste de l'appelant
    H_train = list(H_train)

    sans_frais = [initial_cash]
    brut       = [initial_cash]
    opti       = [initial_cash]
    ref        = [initial_cash]

    w_prev_brut = np.ones(n_dims) / n_dims
    w_prev_opti = np.ones(n_dims) / n_dims
    w_ref = np.ones(n_dims) / n_dims
    frais_taux = 0.005

    for i in range(test_size):
        epsilon = y[i - test_size - 1]
        I_neg = np.diag((epsilon < 0).astype(float))
        H_new = (C @ C.T
                 + A.T @ np.outer(epsilon, epsilon) @ A
                 + B.T @ H_train[-1] @ B
                 + G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)

        mu_next = predict_next_returns_from_returns(y[:i - test_size])

        w_sans = optimal_sharpe_weights(mu_next, H_new)
        w_opti_w = optimal_sharpe_weights(mu_next, H_new,
                                          prev_weights=w_prev_opti,
                                          lambda_tc=frais_taux)

        market_returns = np.exp(y[i - test_size])

        # Sans frais
        v_sans = sans_frais[-1] * np.dot(w_sans, market_returns)

        # Brut : utilise w_sans mais subit les frais
        f_brut = brut[-1] * frais_taux * np.sum(np.abs(w_sans - w_prev_brut))
        v_brut = (brut[-1] - f_brut) * np.dot(w_sans, market_returns)

        # Optimisé : turnover régularisé avec frais réels
        f_opti = opti[-1] * frais_taux * np.sum(np.abs(w_opti_w - w_prev_opti))
        v_opti = (opti[-1] - f_opti) * np.dot(w_opti_w, market_returns)

        # 1/n
        v_ref = ref[-1] * np.dot(w_ref, market_returns)

        sans_frais.append(v_sans)
        brut.append(v_brut)
        opti.append(v_opti)
        ref.append(v_ref)

        w_prev_brut = w_sans
        w_prev_opti = w_opti_w

    return (np.array(sans_frais),
            np.array(brut),
            np.array(opti),
            np.array(ref))


# --- Stratégie 2 : REGU ---
def strat_regu(n_dims, test_size, y, H_train, A, B, C, G, regu_amount):
    """
    Retourne 5 tableaux : (sans_frais, brut, opti, ref, total_investi)
    """
    H_train = list(H_train)

    sans_frais    = [0.0]
    brut          = [0.0]
    opti          = [0.0]
    ref           = [0.0]
    total_investi = [0.0]

    w_prev_brut = np.zeros(n_dims)
    w_prev_opti = np.zeros(n_dims)
    w_ref = np.ones(n_dims) / n_dims
    frais_taux = 0.001

    for i in range(test_size):
        epsilon = y[i - test_size - 1]
        I_neg = np.diag((epsilon < 0).astype(float))
        H_new = (C @ C.T
                 + A.T @ np.outer(epsilon, epsilon) @ A
                 + B.T @ H_train[-1] @ B
                 + G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)

        mu_next = predict_next_returns_from_returns(y[:i - test_size])
        market_returns = np.exp(y[i - test_size])

        w_sans   = optimal_sharpe_weights(mu_next, H_new)
        w_opti_w = optimal_sharpe_weights(mu_next, H_new,
                                          prev_weights=w_prev_opti,
                                          lambda_tc=frais_taux)

        val_avant_sans = sans_frais[-1] + regu_amount
        val_avant_brut = brut[-1]       + regu_amount
        val_avant_opti = opti[-1]       + regu_amount
        val_avant_ref  = ref[-1]        + regu_amount

        v_sans = val_avant_sans * np.dot(w_sans, market_returns)

        f_brut = val_avant_brut * frais_taux * np.sum(np.abs(w_sans - w_prev_brut))
        v_brut = (val_avant_brut - f_brut) * np.dot(w_sans, market_returns)

        f_opti = val_avant_opti * frais_taux * np.sum(np.abs(w_opti_w - w_prev_opti))
        v_opti = (val_avant_opti - f_opti) * np.dot(w_opti_w, market_returns)

        v_ref = val_avant_ref * np.dot(w_ref, market_returns)

        sans_frais.append(v_sans)
        brut.append(v_brut)
        opti.append(v_opti)
        ref.append(v_ref)
        total_investi.append(total_investi[-1] + regu_amount)

        w_prev_brut = w_sans
        w_prev_opti = w_opti_w

    return (np.array(sans_frais),
            np.array(brut),
            np.array(opti),
            np.array(ref),
            np.array(total_investi))


# --- Stratégie 3 : ONLYREGU ---
def strat_only_regu(n_dims, test_size, y, H_train, A, B, C, G, regu_amount):
    """
    On ne rééquilibre que le nouveau cash (pas de rebalancement global).
    Retourne 5 tableaux : (sans_frais, brut, opti, ref, total_investi)
    """
    H_train = list(H_train)

    sans_frais    = [0.0]
    brut          = [0.0]
    opti          = [0.0]
    ref           = [0.0]
    total_investi = [0.0]

    w_prev_sans = np.zeros(n_dims)
    w_prev_brut = np.zeros(n_dims)
    w_prev_opti = np.zeros(n_dims)
    w_ref = np.ones(n_dims) / n_dims
    frais_taux = 0.001

    for i in range(test_size):
        epsilon = y[i - test_size - 1]
        # BUG FIX : terme G réintégré (absent dans la version originale)
        I_neg = np.diag((epsilon < 0).astype(float))
        H_new = (C @ C.T
                 + A.T @ np.outer(epsilon, epsilon) @ A
                 + B.T @ H_train[-1] @ B
                 + G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)

        mu_next = predict_next_returns_from_returns(y[:i - test_size])
        market_returns = np.exp(y[i - test_size])

        # Poids cible pour le nouveau cash uniquement
        w_target = optimal_sharpe_weights(mu_next, H_new)

        def combine(w_old, val_old, w_new, val_new):
            total = val_old + val_new
            if total == 0:
                return np.ones(n_dims) / n_dims
            return (w_old * val_old + w_new * val_new) / total

        # Poids combinés : ancien portefeuille dilué par le nouvel apport
        w_sans   = combine(w_prev_sans, sans_frais[-1], w_target, regu_amount)
        w_brut_w = combine(w_prev_brut, brut[-1],       w_target, regu_amount)
        w_opti_w = combine(w_prev_opti, opti[-1],       w_target, regu_amount)

        val_avant_sans = sans_frais[-1] + regu_amount
        val_avant_brut = brut[-1]       + regu_amount
        val_avant_opti = opti[-1]       + regu_amount
        val_avant_ref  = ref[-1]        + regu_amount

        v_sans = val_avant_sans * np.dot(w_sans, market_returns)

        # BUG FIX : v_brut était dupliqué avec v_opti dans la version originale
        f_brut = val_avant_brut * frais_taux * np.sum(np.abs(w_brut_w - w_prev_brut))
        v_brut = (val_avant_brut - f_brut) * np.dot(w_brut_w, market_returns)

        # Frais uniquement sur l'achat du nouveau cash
        f_opti = regu_amount * frais_taux
        v_opti = (val_avant_opti - f_opti) * np.dot(w_opti_w, market_returns)

        v_ref = val_avant_ref * np.dot(w_ref, market_returns)

        sans_frais.append(v_sans)
        brut.append(v_brut)
        opti.append(v_opti)
        ref.append(v_ref)
        total_investi.append(total_investi[-1] + regu_amount)

        # BUG FIX : mise à jour correcte de TOUS les w_prev
        w_prev_sans = w_sans
        w_prev_brut = w_brut_w
        w_prev_opti = w_opti_w

    return (np.array(sans_frais),
            np.array(brut),
            np.array(opti),
            np.array(ref),
            np.array(total_investi))


# --- Main Execution ---
def main():
    data = np.load("cov_and_modelparams.npz")
    y        = data["y"]
    test_size = int(data["test_size"])
    A, B, C, G = data["A"], data["B"], data["C"], data["G"]
    H_train  = list(data["H_train"])
    n_dims   = int(data["n_dims"])
    dates = data["dates"]
    dates = np.array(dates, dtype="datetime64[ns]")
    dates = dates[-test_size:]

    choice = input("Stratégie (allin/regu/onlyregu) : ").strip().lower()
    initial_total = 10_000

    if choice == "allin":
        sans, brut, opti, ref = strat_all_in(
            n_dims, test_size, y, H_train, A, B, C, G, initial_total
        )
        series = [sans, brut, opti, ref]
        labels = ["Sans frais", "Avec frais (Brut)", "Avec frais (Optimisé)", "1/n"]

    elif choice in ["regu", "onlyregu"]:
        func = strat_regu if choice == "regu" else strat_only_regu
        sans, brut, opti, ref, investi = func(
            n_dims, test_size, y, H_train, A, B, C, G, initial_total / test_size
        )
        series = [sans, brut, opti, ref, investi]
        labels = ["Sans frais", "Avec frais (Brut)", "Avec frais (Optimisé)", "1/n", "Investi"]

    else:
        print("Choix invalide.")
        return

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.figure(figsize=(12, 6))
    for j, (serie, label) in enumerate(zip(series, labels)):
        s = realized_sharpe_from_portfolio_values(serie) if j < 4 else 0
        suffix = f" (S={s:.2f})" if j < 4 else ""
        plt.plot(dates, serie, label=f"{label}{suffix}")

    plt.title(f"Backtest Portfolio - {choice.upper()}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
