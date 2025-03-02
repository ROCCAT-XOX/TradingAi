import os
import argparse
from pathlib import Path

from config.config import config
from utils.alpaca_api import AlpacaAPI
from data.data_loader import DataLoader
from models.trading_agent import TradingAgent
from environment.trading_env import TradingEnvironment
from utils.visualizer import visualize_training, visualize_backtest
from train import train_trading_agent
from backtest import run_backtest
import torch

print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Anzahl GPUs: {torch.cuda.device_count()}")


def parse_args():
    """Kommandozeilenargumente parsen."""
    parser = argparse.ArgumentParser(description="Trading AI mit Reinforcement Learning")

    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'backtest', 'all'],
                        help='Ausführungsmodus (default: all)')

    parser.add_argument('--symbol', type=str,
                        help='Trading-Symbol (z.B. BTC/USD oder AAPL)')

    parser.add_argument('--episodes', type=int,
                        help='Anzahl der Trainingsepisoden')

    parser.add_argument('--config', type=str,
                        help='Pfad zur Konfigurationsdatei')

    parser.add_argument('--lr', type=float,
                        help='Lernrate für das Training')

    parser.add_argument('--gamma', type=float,
                        help='Discount-Faktor für zukünftige Belohnungen')

    parser.add_argument('--use-lstm', action='store_true',
                        help='LSTM-Netzwerk verwenden')

    parser.add_argument('--hidden-dim', type=int,
                        help='Dimension der versteckten Schichten')

    parser.add_argument('--window-size', type=int,
                        help='Größe des Beobachtungsfensters')

    return parser.parse_args()


def main():
    """Hauptfunktion des Programms."""
    # Argumente parsen
    args = parse_args()

    # Konfiguration laden (ggf. mit benutzerdefiniertem Pfad)
    if args.config:
        config_instance = config.__class__(args.config)
    else:
        config_instance = config

    # Symbol überschreiben, falls angegeben
    symbol = args.symbol or config_instance.get("trading", "symbol", "BTC/USD")

    # Episodenanzahl überschreiben, falls angegeben
    episodes = args.episodes or config_instance.get("training", "episodes", 100)

    # Weitere Parameter aus Argumenten oder Konfiguration
    lr = args.lr or config_instance.get("model", "learning_rate", 0.0003)
    gamma = args.gamma or config_instance.get("model", "gamma", 0.99)
    use_lstm = args.use_lstm or config_instance.get("model", "use_lstm", False)
    hidden_dim = args.hidden_dim or config_instance.get("model", "hidden_dim", 128)
    window_size = args.window_size or config_instance.get("model", "window_size", 10)

    # Verzeichnisse erstellen
    save_path = config_instance.get("training", "save_path", "saved_models")
    results_path = config_instance.get("visualization", "save_path", "results")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    print(f"=== Trading AI für {symbol} ===")
    print(f"Parameter: LR={lr}, Gamma={gamma}, LSTM={use_lstm}, Hidden Dim={hidden_dim}")
    print(f"Window Size={window_size}, Episodes={episodes}")

    # Alpaca API initialisieren
    api = AlpacaAPI()

    # Daten laden
    data_loader = DataLoader(api)
    df = data_loader.load_historical_data(symbol)

    if df.empty:
        print(f"Keine Daten für {symbol} gefunden. Beende Programm.")
        return

    # Trading Environment initialisieren
    env_params = {
        'initial_balance': config_instance.get("trading", "initial_balance", 10000),
        'commission': config_instance.get("trading", "commission", 0.001),
        'window_size': window_size
    }

    env = TradingEnvironment(df, **env_params)

    print(f"Observation dimension: {env.obs_dim}")

    # Agent initialisieren mit korrekter Dimension
    agent_params = {
        'input_dim': env.obs_dim,
        'n_actions': 3,
        'lr': lr,
        'gamma': gamma,
        'use_lstm': use_lstm,
        'hidden_dim': hidden_dim
    }

    agent = TradingAgent(**agent_params)

    # Modelldatei
    model_suffix = "lstm" if use_lstm else "mlp"
    model_path = Path(save_path) / f"{symbol.replace('/', '_')}_{model_suffix}_model.pth"

    # Training durchführen
    if args.mode in ['train', 'all']:
        print("\n=== Starte Training ===")
        training_params = {
            'episodes': episodes,
            'eval_every': config_instance.get("training", "eval_every", 10)
        }

        try:
            history = train_trading_agent(env, agent, model_path=model_path, **training_params)

            # Ergebnisse visualisieren
            visualize_training(
                history,
                symbol,
                save_path=Path(results_path) / f"{symbol.replace('/', '_')}_training.png"
            )
        except Exception as e:
            print(f"Fehler während des Trainings: {e}")
            import traceback
            traceback.print_exc()
            return

    # Backtesting durchführen
    if args.mode in ['backtest', 'all']:
        print("\n=== Starte Backtesting ===")
        # Modell laden, falls vorhanden
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Modell geladen: {model_path}")
        else:
            if args.mode == 'backtest':
                print(f"Kein gespeichertes Modell gefunden: {model_path}")
                print("Bitte führe zuerst das Training durch oder gib einen gültigen Modellpfad an.")
                return

        try:
            results = run_backtest(df, agent, env_params)

            # Ergebnisse visualisieren
            visualize_backtest(
                df,
                results,
                symbol,
                save_path=Path(results_path) / f"{symbol.replace('/', '_')}_backtest.png"
            )
        except Exception as e:
            print(f"Fehler während des Backtestings: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Programm abgeschlossen ===")


if __name__ == "__main__":
    main()