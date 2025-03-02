import streamlit as st

# Page config MUST be the first Streamlit command
st.set_page_config(page_title="Trading AI Dashboard", layout="wide")

import os
import sys
import pandas as pd
import numpy as np

# Importiere Torch mit zusätzlicher Fehlerbehandlung
try:
    import torch
except ImportError as e:
    st.error(f"Torch-Import fehlgeschlagen: {e}")
    st.error("Stellen Sie sicher, dass PyTorch korrekt installiert ist.")
    torch = None

# Importiere lokale Module
try:
    from config.config import config
    from utils.alpaca_api import AlpacaAPI
    from data.data_loader import DataLoader
    from models.trading_agent import TradingAgent
    from environment.trading_env import TradingEnvironment
    from utils.visualizer import visualize_training, visualize_backtest
    from train import train_trading_agent, evaluate_agent
    from backtest import run_backtest
except ImportError as e:
    st.error(f"Fehler beim Importieren der lokalen Module: {e}")
    st.error("Stellen Sie sicher, dass sich alle Module im Python-Pfad befinden.")
    # Dummy-Funktionen für Fehlerfall
    config = None
    AlpacaAPI = None
    DataLoader = None
    TradingAgent = None
    TradingEnvironment = None
    visualize_training = None
    train_trading_agent = None
    evaluate_agent = None
    run_backtest = None


def check_dependencies():
    """Überprüft alle kritischen Abhängigkeiten"""
    dependencies = [
        ("Torch", torch is not None),
        ("Config", config is not None),
        ("AlpacaAPI", AlpacaAPI is not None),
        ("DataLoader", DataLoader is not None),
        ("TradingAgent", TradingAgent is not None),
        ("TradingEnvironment", TradingEnvironment is not None),
    ]

    all_ok = True
    for name, status in dependencies:
        if not status:
            st.error(f"{name} konnte nicht geladen werden!")
            all_ok = False

    return all_ok


def main():
    # Überprüfe Systemvoraussetzungen
    if sys.version_info >= (3, 12):
        st.warning("Python 3.12 kann Kompatibilitätsprobleme verursachen. Empfohlen: Python 3.10 oder 3.11")

    # Überprüfe kritische Abhängigkeiten
    if not check_dependencies():
        st.error("Nicht alle erforderlichen Abhängigkeiten sind verfügbar.")
        return

    st.title("🤖 Trading AI Dashboard")

    # Seitenleiste für Konfiguration
    st.sidebar.header("Trading Konfiguration")

    # Symbol Auswahl
    available_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AAPL", "MSFT", "GOOGL"]

    # Sicheres Laden der Konfiguration
    try:
        default_symbol = config.get("trading", "symbol", "BTC/USD")
    except Exception:
        default_symbol = "BTC/USD"

    symbol = st.sidebar.selectbox("Wähle Trading Symbol", available_symbols,
                                  index=available_symbols.index(default_symbol))

    # Zeitrahmen Auswahl
    timeframes = ["1Min", "5Min", "15Min", "1H", "4H", "1D"]

    # Sicheres Laden des Zeitrahmens
    try:
        default_timeframe = config.get("trading", "timeframe", "1D")
    except Exception:
        default_timeframe = "1D"

    timeframe = st.sidebar.selectbox("Wähle Zeitrahmen", timeframes,
                                     index=timeframes.index(default_timeframe))

    # Training Konfiguration
    col1, col2 = st.sidebar.columns(2)
    with col1:
        # Sicheres Laden der Episoden
        try:
            default_episodes = config.get("training", "episodes", 100)
        except Exception:
            default_episodes = 100

        episodes = st.number_input("Episoden", min_value=10, max_value=1000,
                                   value=default_episodes)
    with col2:
        # Sicheres Laden der LSTM-Option
        try:
            default_lstm = config.get("model", "use_lstm", False)
        except Exception:
            default_lstm = False

        use_lstm = st.checkbox("LSTM Netzwerk", value=default_lstm)

    # Hyperparameter
    st.sidebar.subheader("Hyperparameter")

    # Sicheres Laden von Hyperparametern
    try:
        default_lr = config.get("model", "learning_rate", 0.0003)
        default_gamma = config.get("model", "gamma", 0.99)
    except Exception:
        default_lr = 0.0003
        default_gamma = 0.99

    lr = st.sidebar.slider("Lernrate", 0.0001, 0.01,
                           value=default_lr,
                           format="%.4f")
    gamma = st.sidebar.slider("Discount Faktor", 0.9, 0.999,
                              value=default_gamma,
                              format="%.3f")

    # Buttons für Aktionen
    col1, col2, col3 = st.columns(3)

    with col1:
        train_button = st.button("Training starten")
    with col2:
        backtest_button = st.button("Backtesting durchführen")
    with col3:
        cuda_info = st.button("CUDA Info")

    # CUDA Informationen
    if cuda_info:
        st.subheader("🖥️ CUDA Informationen")
        try:
            if torch is not None:
                st.write(f"PyTorch Version: {torch.__version__}")
                st.write(f"CUDA verfügbar: {torch.cuda.is_available()}")

                if torch.cuda.is_available():
                    try:
                        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
                        st.write(f"CUDA Version: {torch.version.cuda}")
                        st.write(
                            f"Gesamt-GPU-Speicher: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
                    except Exception as gpu_error:
                        st.warning(f"Fehler beim Abrufen von GPU-Details: {gpu_error}")
                else:
                    st.warning("Keine CUDA-fähige GPU gefunden.")
            else:
                st.error("PyTorch nicht geladen.")
        except Exception as cuda_error:
            st.error(f"Fehler bei CUDA-Informationen: {cuda_error}")

    # Ergebnisse Verzeichnisse
    try:
        results_path = config.get("visualization", "save_path", "results")
        save_path = config.get("training", "save_path", "saved_models")
    except Exception:
        results_path = "results"
        save_path = "saved_models"

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Training
    if train_button:
        with st.spinner('Training läuft...'):
            try:
                # API und Daten laden
                api = AlpacaAPI()
                data_loader = DataLoader(api)
                df = data_loader.load_historical_data(symbol, timeframe=timeframe)

                if df.empty:
                    st.error(f"Keine Daten für {symbol} gefunden.")
                    return

                # Trading Environment initialisieren
                env_params = {
                    'initial_balance': config.get("trading", "initial_balance", 10000),
                    'commission': config.get("trading", "commission", 0.001),
                    'window_size': config.get("model", "window_size", 10)
                }
                env = TradingEnvironment(df, **env_params)

                # Agent initialisieren
                agent_params = {
                    'input_dim': env.obs_dim,
                    'n_actions': 3,
                    'lr': lr,
                    'gamma': gamma,
                    'use_lstm': use_lstm,
                    'hidden_dim': config.get("model", "hidden_dim", 128)
                }
                agent = TradingAgent(**agent_params)

                # Modellpfad
                model_filename = f"{symbol.replace('/', '_')}_{'lstm' if use_lstm else 'mlp'}_model.pth"
                model_path = os.path.join(save_path, model_filename)

                # Training durchführen
                history = train_trading_agent(
                    env,
                    agent,
                    episodes=episodes,
                    model_path=model_path
                )

                # Performance visualisieren
                training_plot = os.path.join(results_path, f"{symbol.replace('/', '_')}_training.png")
                visualize_training(history, symbol, save_path=training_plot)

                # Plots anzeigen
                st.subheader("🏋️ Trainingsergebnisse")
                st.image(training_plot)

                # Evaluierung
                eval_return = evaluate_agent(env, agent)
                st.metric("Evaluierungs-Return", f"{eval_return:.2f}%")

            except Exception as e:
                st.error(f"Fehler während des Trainings: {e}")
                import traceback
                st.error(traceback.format_exc())

    # Backtesting
    if backtest_button:
        with st.spinner('Backtesting läuft...'):
            try:
                # API und Daten laden
                api = AlpacaAPI()
                data_loader = DataLoader(api)
                df = data_loader.load_historical_data(symbol, timeframe=timeframe)

                if df.empty:
                    st.error(f"Keine Daten für {symbol} gefunden.")
                    return

                # Modellpfad
                model_filename = f"{symbol.replace('/', '_')}_{'lstm' if use_lstm else 'mlp'}_model.pth"
                model_path = os.path.join(save_path, model_filename)

                # Trading Environment und Agent initialisieren
                env_params = {
                    'initial_balance': config.get("trading", "initial_balance", 10000),
                    'commission': config.get("trading", "commission", 0.001),
                    'window_size': config.get("model", "window_size", 10)
                }
                env = TradingEnvironment(df, **env_params)

                agent_params = {
                    'input_dim': env.obs_dim,
                    'n_actions': 3,
                    'use_lstm': use_lstm
                }
                agent = TradingAgent(**agent_params)

                # Modell laden
                if not agent.load(model_path):
                    st.error("Kein trainiertes Modell gefunden. Führen Sie zuerst das Training durch.")
                    return

                # Backtesting durchführen
                results = run_backtest(df, agent, env_params)

                # Plots für Backtest
                backtest_plot = os.path.join(results_path, f"{symbol.replace('/', '_')}_backtest.png")
                backtest_metrics_plot = os.path.join(results_path, f"{symbol.replace('/', '_')}_backtest_metrics.png")
                visualize_backtest(df, results, symbol, save_path=backtest_plot)

                # Plots anzeigen
                st.subheader("📊 Backtesting-Ergebnisse")

                # Performance-Metriken
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strategie Return", f"{results.get('return', 0):.2f}%")
                with col2:
                    st.metric("Buy & Hold Return", f"{results.get('buy_hold_return', 0):.2f}%")
                with col3:
                    st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2f}%")

                # Backtesting Plots
                st.image(backtest_plot)
                st.image(backtest_metrics_plot)

            except Exception as e:
                st.error(f"Fehler während des Backtestings: {e}")
                import traceback
                st.error(traceback.format_exc())


if __name__ == "__main__":
    main()