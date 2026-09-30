import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from src.models.bekk_gjr import (
    fit_bekk_gjr, MGARCH_GJR, compute_bekk_gjr_covariances
)
from src.backtest.engine import (
    strat_all_in, strat_regu, strat_only_regu,
    realized_sharpe_from_portfolio_values
)
try:
    from src.data.tickers import TICKERS_DICT, TICKER_NAMES
except ImportError:
    TICKERS_DICT = {}
    TICKER_NAMES = []

# -------------------------
# CONFIG PAGE
# -------------------------
st.set_page_config(layout="wide")

# -------------------------
# LANGUE
# -------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "FR"

with st.sidebar:
    if st.button("FR / EN"):
        st.session_state.lang = "EN" if st.session_state.lang == "FR" else "FR"

lang = st.session_state.lang

TEXT = {
    "FR": {
        "title": "Backtest GJR-BEKK-GARCH yfinance - Sauvegarde auto",
        "tickers": "Entrez les tickers séparés par une virgule",
        "ticker_search_header": "Recherche de ticker",
        "ticker_search_input": "Cherchez une entreprise pour connaître son ticker",
        "ticker_search_matches": "Correspondances trouvées",
        "ticker_search_none": "Aucune correspondance trouvée",
        "capital": "Capital initial",
        "ratio": "Proportion des données pour backtest",
        "start": "Début des données",
        "end": "Fin des données",
        "strategy": "Choisissez la stratégie",
        "train": "Lancer l'entrainement",
        "backtest": "Lancer le backtest",
        "results": "Résultats backtest",
        "params": "Paramètres",
        "info": "Configurez les paramètres dans la barre latérale puis lancez l'entraînement et le backtest.",
        "intro": (
            "Cette application permet de backtester des stratégies de portefeuille basées sur un modèle "
            "GJR-BEKK-GARCH estimé sur des données financières. "
            "Vous pouvez comparer différentes approches d'investissement (all-in, régulier) et analyser leur performance en termes de risque et rendement."
            "Frais = 0.5%/trade et Taux sans Risque = 3%/an"
        ),
        "expander": "Explication des stratégies",
        "data_loading": "Téléchargement des données...",
        "model_estimation": "Estimation du modèle BEKK-GJR...",
        "model_application": "Application du modèle BEKK-GJR...",
        "data_loaded": "Données téléchargées",
        "model_done": "Modèle estimé",
        "backtest_done": "Backtest terminé",
        "days": "Jours",
        "portfolio": "Valeur du portefeuille",
        "invested": "Argent investi",
        "opt_no_fees": "Portefeuille optimisé sans frais",
        "opt_with_fees": "Portefeuille optimisé avec frais",
        "opt_fees": "Portefeuille optimisant les frais",
        "benchmark": "Portefeuille 1/n",
        "title_allin": "Backtest : stratégie All-in",
        "title_regu": "Backtest : stratégie REGU",
        "title_onlyregu": "Backtest : stratégie ONLYREGU",
        "no_file": "Aucun fichier backtest trouvé. Lancez d'abord l'entraînement.",
    },
    "EN": {
        "title": "GJR-BEKK-GARCH Backtest yfinance - Auto save",
        "tickers": "Enter tickers separated by commas",
        "ticker_search_header": "Ticker Search",
        "ticker_search_input": "Search a company to find its ticker",
        "ticker_search_matches": "Matches found",
        "ticker_search_none": "No matches found",
        "capital": "Initial capital",
        "ratio": "Backtest data proportion",
        "start": "Start date",
        "end": "End date",
        "strategy": "Choose strategy",
        "train": "Run training",
        "backtest": "Run backtest",
        "results": "Backtest results",
        "params": "Parameters",
        "info": "Configure parameters in the sidebar then run training and backtest.",
        "intro": (
            "This application allows you to backtest portfolio strategies based on a GJR-BEKK-GARCH model "
            "estimated from financial data. "
            "You can compare different investment approaches (all-in, periodic) "
            "and analyze their performance in terms of risk and return."
            "Fees = 0.5%/trade et Risk free return = 3%/year"
        ),
        "expander": "Strategy explanation",
        "data_loading": "Downloading data...",
        "model_estimation": "Estimating BEKK-GJR model...",
        "model_application": "Applying BEKK-GJR model...",
        "data_loaded": "Data loaded",
        "model_done": "Model estimated",
        "backtest_done": "Backtest completed",
        "days": "Days",
        "portfolio": "Portfolio value",
        "invested": "Invested capital",
        "opt_no_fees": "Optimized portfolio (no fees)",
        "opt_with_fees": "Optimized portfolio (with fees)",
        "opt_fees": "Fees-optimized portfolio",
        "benchmark": "Equal weight portfolio",
        "title_allin": "Backtest: All-in strategy",
        "title_regu": "Backtest: Regular investment strategy",
        "title_onlyregu": "Backtest: Only regular investment",
        "no_file": "No backtest file found. Please run training first.",
    },
}

st.title(TEXT[lang]["title"])
st.markdown(TEXT[lang]["intro"])
st.info(TEXT[lang]["info"])

# -------------------------
# VALEURS PAR DÉFAUT
# -------------------------
_default_tickers       = ["AI.PA", "CS.PA"]
_default_capital       = 10_000
_default_start         = pd.Timestamp("2010-01-01")
_default_end           = pd.Timestamp("2025-12-06")
_default_test_ratio    = 0.15
_default_n_dims        = 2

# -------------------------
# BUG FIX : chargement du fichier NPZ avec gestion d'absence
# -------------------------
_npz_loaded = False
try:
    backtest = np.load("backtest_results.npz", allow_pickle=True)

    tickers        = list(backtest["tickers"])
    n_dims         = int(backtest["n_dims"])          # BUG FIX : cast en int
    y              = backtest["y"]
    test_size      = int(backtest["test_size"])       # BUG FIX : cast en int
    H_train        = list(backtest["H_train"])
    C              = backtest["C"]
    A              = backtest["A"]
    B              = backtest["B"]
    G              = backtest["G"]
    initial_capital = float(backtest["initial_capital"])
    start_date     = pd.to_datetime(str(backtest["start_date"]))
    backtest_start_date = pd.to_datetime(str(backtest["backtest_start_date"]))
    end_date       = pd.to_datetime(str(backtest["end_date"]))

    allin_opt            = backtest["allin_opt"]
    allin_opt_puis_frais = backtest["allin_opt_puis_frais"]
    allin_opt_avec_frais = backtest["allin_opt_avec_frais"]
    allin_ref            = backtest["allin_ref"]

    regu_opt            = backtest["regu_opt"]
    regu_opt_puis_frais = backtest["regu_opt_puis_frais"]
    regu_opt_avec_frais = backtest["regu_opt_avec_frais"]
    regu_ref            = backtest["regu_ref"]

    only_regu_opt            = backtest["only_regu_opt"]
    only_regu_opt_puis_frais = backtest["only_regu_opt_puis_frais"]
    only_regu_opt_avec_frais = backtest["only_regu_opt_avec_frais"]
    only_regu_ref            = backtest["only_regu_ref"]

    _npz_loaded = True

except FileNotFoundError:
    st.warning(TEXT[lang]["no_file"])
    tickers         = _default_tickers
    n_dims          = _default_n_dims
    y               = np.array([])
    test_size       = 100
    H_train         = []
    C = A = B = G   = np.eye(_default_n_dims)
    initial_capital = _default_capital
    start_date      = _default_start
    end_date        = _default_end
    # Tableaux vides pour éviter les NameError dans la section affichage
    allin_opt = allin_opt_puis_frais = allin_opt_avec_frais = allin_ref = np.array([0.0])
    regu_opt  = regu_opt_puis_frais  = regu_opt_avec_frais  = regu_ref  = np.array([0.0])
    only_regu_opt = only_regu_opt_puis_frais = only_regu_opt_avec_frais = only_regu_ref = np.array([0.0])

# -------------------------
# SIDEBAR PARAMÈTRES
# -------------------------
with st.sidebar:
    st.header(TEXT[lang]["ticker_search_header"])

    search_input = st.text_input(TEXT[lang]["ticker_search_input"])
    if search_input and TICKER_NAMES:
        matches = [name for name in TICKER_NAMES if search_input.lower() in name.lower()]
        if matches:
            selected = st.selectbox(TEXT[lang]["ticker_search_matches"], matches)
            st.write(f"{selected} → {TICKERS_DICT[selected]}")
        else:
            st.write(TEXT[lang]["ticker_search_none"])

    st.header(TEXT[lang]["params"])

    tickers_input = st.text_input(
        TEXT[lang]["tickers"],
        value=",".join(tickers),
    )
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    n_dims  = len(tickers)

    initial_capital = st.number_input(
        TEXT[lang]["capital"],
        value=int(initial_capital),
        step=1_000,
    )

    test_ratio = st.slider(
        TEXT[lang]["ratio"],
        0.05, 0.30,
        value=float(test_size) / len(y) if len(y) > 0 else _default_test_ratio,
    )

    start_date = st.date_input(TEXT[lang]["start"],value=start_date,min_value=pd.Timestamp("1900-01-01"),max_value=pd.Timestamp("2026-12-31") )
    end_date = st.date_input(TEXT[lang]["end"],value=end_date,min_value=pd.Timestamp("1900-01-01"),max_value=pd.Timestamp("2026-12-31") )

    strategie = st.selectbox(
        TEXT[lang]["strategy"],
        ["allin", "regu", "onlyregu"],
    )

    st.markdown("---")
    train_button   = st.button(TEXT[lang]["train"])
    backtest_button = st.button(TEXT[lang]["backtest"])

# -------------------------
# EXPLICATION
# -------------------------
with st.expander(TEXT[lang]["expander"]):
    if lang == "FR":
        st.write(
            "**allin** : investissement total initial\n\n"
            "**regu** : investissement progressif avec rééquilibrage complet\n\n"
            "**onlyregu** : le nouveau cash seul est investi aux poids optimaux "
            "(pas de rebalancement global du portefeuille existant)"
        )
    else:
        st.write(
            "**allin** : full initial investment\n\n"
            "**regu** : progressive investment with full rebalancing\n\n"
            "**onlyregu** : only the new cash is invested at optimal weights "
            "(no global rebalancing of the existing portfolio)"
        )

# -------------------------
# TRAIN
# -------------------------
if train_button:

    with st.spinner(TEXT[lang]["data_loading"]):
        data = yf.download(tickers, start=start_date, end=end_date,
                           auto_adjust=True)["Close"]
        # Gestion du cas mono-ticker (DataFrame → Series → reshape)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        y_df = np.log(data).diff().dropna()
        backtest_start_date = y_df.index[-test_size]
        y_matrix = y_df.values
        test_size = int(test_ratio * len(y_matrix))

    # --- Validation avant tout calcul ---
    if len(y_matrix) == 0:
        st.error("Aucune donnée téléchargée. Vérifiez les tickers et les dates.")
        st.stop()

    if test_size == 0:
        st.error(
            f"test_size = 0 : le ratio {test_ratio:.0%} × {len(y_matrix)} observations "
            f"= 0 jours de test. Augmentez le ratio ou élargissez la plage de dates."
        )
        st.stop()

    # BUG FIX : y_matrix[:-0] == y_matrix[:0] == [] en numpy/Python
    #           On utilise un slice explicite avec test_size > 0 garanti
    y_train = y_matrix[: len(y_matrix) - test_size]

    min_obs_required = max(50, n_dims * 10)
    if len(y_train) < min_obs_required:
        st.error(
            f"Trop peu d'observations en entraînement ({len(y_train)}). "
            f"Minimum requis : {min_obs_required}. Élargissez la plage de dates."
        )
        st.stop()

    st.success(f"{TEXT[lang]['data_loaded']} : {len(y_train)} train + {test_size} test, {n_dims} tickers")

    with st.spinner(TEXT[lang]["model_estimation"]):
        try:
            result = fit_bekk_gjr(y_train, n_dims)
            model  = MGARCH_GJR.from_params(result.x, n_dims)
            H_train_list = list(
                compute_bekk_gjr_covariances(y_train, model.C, model.A, model.B, model.G)
            )
        except Exception as e:
            st.error(f"Erreur lors de l'estimation du modèle BEKK-GJR : {e}")
            st.stop()

    st.success(TEXT[lang]["model_done"])

    with st.spinner(TEXT[lang]["model_application"]):
        try:
            # BUG FIX : test_size garanti > 0 (vérifié plus haut), division sûre
            regu_amount = initial_capital / test_size
        
            # Backtest ALL-IN
            allin_opt, allin_opt_puis_frais, allin_opt_avec_frais, allin_ref = strat_all_in(
                n_dims=n_dims,
                test_size=test_size,
                y=y_matrix,
                H_train=H_train_list,
                A=model.A, B=model.B, C=model.C, G=model.G,
                initial_cash=initial_capital,
            )
        
            # Backtest REGU
            regu_opt, regu_opt_puis_frais, regu_opt_avec_frais, regu_ref, _ = strat_regu(
                n_dims=n_dims,
                test_size=test_size,
                y=y_matrix,
                H_train=H_train_list,
                A=model.A, B=model.B, C=model.C, G=model.G,
                regu_amount=regu_amount,
            )
        
            # Backtest ONLYREGU
            (only_regu_opt, only_regu_opt_puis_frais,
             only_regu_opt_avec_frais, only_regu_ref, _) = strat_only_regu(
                n_dims=n_dims,
                test_size=test_size,
                y=y_matrix,
                H_train=H_train_list,
                A=model.A, B=model.B, C=model.C, G=model.G,
                regu_amount=regu_amount,
            )
        
            save_path = "backtest_results.npz"
            np.savez_compressed(
            save_path,
            tickers=tickers,
            n_dims=n_dims,
            y=y_matrix,
            test_size=test_size,
            H_train=H_train_list,
            C=model.C, A=model.A, B=model.B, G=model.G,
            initial_capital=initial_capital,
            start_date=str(start_date),
            backtest_start_date=str(y_df.index[-test_size]),
            end_date=str(end_date),
            allin_opt=allin_opt,
            allin_opt_puis_frais=allin_opt_puis_frais,
            allin_opt_avec_frais=allin_opt_avec_frais,
            allin_ref=allin_ref,
            regu_opt=regu_opt,
            regu_opt_puis_frais=regu_opt_puis_frais,
            regu_opt_avec_frais=regu_opt_avec_frais,
            regu_ref=regu_ref,
            only_regu_opt=only_regu_opt,
            only_regu_opt_puis_frais=only_regu_opt_puis_frais,
            only_regu_opt_avec_frais=only_regu_opt_avec_frais,
            only_regu_ref=only_regu_ref,
            )
            #st.success(f"Résultats sauvegardés dans {save_path}")
            #with open(save_path, "rb") as f:
            #    st.download_button("Télécharger le fichier .npz", data=f, file_name=save_path)
        except Exception as e:
            st.warning(f"Sauvegarde échouée ({{e}}), résultats affichables ci-dessous.")

    st.success(TEXT[lang]["backtest_done"])

# -------------------------
# AFFICHAGE BACKTEST
# -------------------------
if backtest_button:

    if not _npz_loaded and not train_button:
        st.error(TEXT[lang]["no_file"])
    else:
        st.header(TEXT[lang]["results"])
        st.subheader(f"Dates Backtest: {pd.to_datetime(backtest_start_date).date()} -> {pd.to_datetime(end_date).date()}. Tickers: {', '.join(tickers)}")
        fig, ax = plt.subplots(figsize=(10, 6))

        if strategie == "allin":
            sharpe_opt            = realized_sharpe_from_portfolio_values(allin_opt)
            sharpe_opt_puis_frais = realized_sharpe_from_portfolio_values(allin_opt_puis_frais)
            sharpe_opt_avec_frais = realized_sharpe_from_portfolio_values(allin_opt_avec_frais)
            sharpe_ref            = realized_sharpe_from_portfolio_values(allin_ref)

            ax.plot(allin_opt,            label=f"{TEXT[lang]['opt_no_fees']}, S={sharpe_opt:.2f}")
            ax.plot(allin_opt_puis_frais, label=f"{TEXT[lang]['opt_with_fees']}, S={sharpe_opt_puis_frais:.2f}")
            ax.plot(allin_opt_avec_frais, label=f"{TEXT[lang]['opt_fees']}, S={sharpe_opt_avec_frais:.2f}")
            ax.plot(allin_ref,            label=f"{TEXT[lang]['benchmark']}, S={sharpe_ref:.2f}", linestyle="--")
            ax.set_title(TEXT[lang]["title_allin"])

        elif strategie == "regu":
            n_pts = len(regu_opt)
            total = [i * initial_capital / max(n_pts - 1, 1) for i in range(n_pts)]
            ax.plot(regu_opt,            label=TEXT[lang]["opt_no_fees"])
            ax.plot(regu_opt_puis_frais, label=TEXT[lang]["opt_with_fees"])
            ax.plot(regu_opt_avec_frais, label=TEXT[lang]["opt_fees"])
            ax.plot(regu_ref,            label=TEXT[lang]["benchmark"], linestyle="--")
            ax.plot(total,               label=TEXT[lang]["invested"],  linestyle=":")
            ax.set_title(TEXT[lang]["title_regu"])

        else:  # onlyregu
            n_pts = len(only_regu_opt)
            total = [i * initial_capital / max(n_pts - 1, 1) for i in range(n_pts)]
            ax.plot(only_regu_opt,            label=TEXT[lang]["opt_no_fees"])
            ax.plot(only_regu_opt_puis_frais, label=TEXT[lang]["opt_with_fees"])
            ax.plot(only_regu_opt_avec_frais, label=TEXT[lang]["opt_fees"])
            ax.plot(only_regu_ref,            label=TEXT[lang]["benchmark"], linestyle="--")
            ax.plot(total,                    label=TEXT[lang]["invested"],  linestyle=":")
            ax.set_title(TEXT[lang]["title_onlyregu"])

        ax.set_xlabel(TEXT[lang]["days"])
        ax.set_ylabel(TEXT[lang]["portfolio"])
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
