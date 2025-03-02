import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import config
from utils.alpaca_api import AlpacaAPI
from data.data_loader import DataLoader
from models.trading_agent import TradingAgent
from environment.trading_env import TradingEnvironment
from utils.visualizer import visualize_training


def parse_args():
    """Kommandozeilenargumente parsen."""
    parser = argparse.ArgumentParser(description="Trading AI Training")

    parser.add_argument('--symbol', type=str,
                        help='Trading-Symbol (z.B. BTC/USD oder AAPL)')

    parser.add_argument('--episodes', type=int, default=100,
                        help='Anzahl der Trainingsepisoden')

    parser.add_argument('--window', type=int, default=10,
                        help='Größe des Beobachtungsfensters')

    parser.add_argument('--balance', type=float, default=10000,
                        help='Anfangskapital')

    parser.add_argument('--commission', type=float, default=0.001,
                        help='Handelsgebühr (z.B. 0.001 für 0.1%)')

    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Lernrate')

    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount-Faktor')

    parser.add_argument('--use-lstm', action='store_true',
                        help='LSTM-Netzwerk verwenden')

    parser.add_argument('--eval-every', type=int, default=10,
                        help='Nach wie vielen Episoden Evaluierung durchgeführt wird')

    parser.add_argument('--save-path', type=str, default='saved_models',
                        help='Pfad zum Speichern des Modells')

    return parser.parse_args()


def train_trading_agent(env, agent, episodes=100, eval_every=10, model_path=None):
    """
    Trainiert den Trading-Agenten und verfolgt Fortschritte.

    Args:
        env: Trading-Umgebung
        agent: Trading-Agent
        episodes: Anzahl der Trainingsepisoden
        eval_every: Nach wie vielen Episoden Evaluierung durchgeführt wird
        model_path: Pfad zum Speichern des besten Modells

    Returns:
        dict: Trainingsverlauf
    """
    training_history = {
        'episode_rewards': [],
        'portfolio_values': [],
        'eval_returns': []
    }

    best_return = -np.inf

    for episode in tqdm(range(episodes), desc="Training RL-Agent"):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Wähle Aktion
            action = agent.select_action(state, training=True)

            # Führe Aktion aus
            next_state, reward, done, _, info = env.step(action)

            # Speichere Belohnung
            agent.store_reward(reward)
            episode_reward += reward

            # Wenn Episode beendet oder Fenster voll, führe Policy-Update durch
            if done:
                agent.update(done=True)
            # Ansonsten alle 5 Schritte updaten (auch möglich: nach jedem Schritt)
            elif (info['step'] - env.window_size) % 5 == 0:
                agent.update(next_state, done=False)

            # Update Zustand
            state = next_state

        # Speichere Episodendaten
        training_history['episode_rewards'].append(episode_reward)
        training_history['portfolio_values'].append(env.portfolio_values)

        # Logge Fortschritt
        final_value = env.portfolio_values[-1]
        returns = (final_value / env.initial_balance - 1) * 100

        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.4f}, "
              f"Endwert: ${final_value:.2f}, Return: {returns:.2f}%")

        # Evaluierung nach festgelegten Intervallen
        if (episode + 1) % eval_every == 0:
            eval_return = evaluate_agent(env, agent, episodes=3)
            training_history['eval_returns'].append(eval_return)

            print(f"Evaluierung nach Episode {episode + 1}: Return = {eval_return:.2f}%")

            # Speichere bestes Modell
            if eval_return > best_return and model_path is not None:
                best_return = eval_return
                agent.save(model_path)
                print(f"Neues bestes Modell gespeichert mit Return: {eval_return:.2f}%")

    return training_history


def evaluate_agent(env, agent, episodes=5):
    """
    Evaluiert den Agenten ohne Exploration.

    Args:
        env: Trading-Umgebung
        agent: Trading-Agent
        episodes: Anzahl der Evaluierungsepisoden

    Returns:
        float: Durchschnittlicher Return
    """
    returns = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # Wähle beste Aktion (ohne Exploration)
            action = agent.select_action(state, training=False)

            # Führe Aktion aus
            next_state, _, done, _, _ = env.step(action)
            state = next_state

        # Berechne Return
        final_return = (env.portfolio_values[-1] / env.initial_balance - 1) * 100
        returns.append(final_return)

    return np.mean(returns)


def main():
    """Hauptfunktion für das Training."""
    args = parse_args()

    # Symbol aus Argumenten oder Config
    symbol = args.symbol or config.get("trading", "symbol", "BTC/USD")

    # API initialisieren und Daten laden
    api = AlpacaAPI()
    data_loader = DataLoader(api)
    df = data_loader.load_historical_data(symbol)

    if df.empty:
        print(f"Keine Daten für {symbol} gefunden. Beende Programm.")
        return

    # Trading Environment initialisieren
    env_params = {
        'initial_balance': args.balance,
        'commission': args.commission,
        'window_size': args.window
    }

    env = TradingEnvironment(df, **env_params)

    # Agent initialisieren
    agent_params = {
        'input_dim': env.obs_dim,  # <-- Das sollte die korrekte Dimension verwenden
        'n_actions': 3,
        'lr': args.lr,
        'gamma': args.gamma,
        'use_lstm': args.use_lstm,
        'hidden_dim': config.get("model", "hidden_dim", 128)
    }

    agent = TradingAgent(**agent_params)

    # Pfad für Modell erstellen
    model_filename = f"{symbol.replace('/', '_')}_{'lstm' if args.use_lstm else 'mlp'}_model.pth"
    model_path = os.path.join(args.save_path, model_filename)

    # Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs(args.save_path, exist_ok=True)

    print(f"=== Starte Training für {symbol} ===")
    print(f"Modell wird gespeichert unter: {model_path}")
    print(f"Beobachtungsfenster: {args.window}, Episoden: {args.episodes}")
    print(f"Netzwerktyp: {'LSTM' if args.use_lstm else 'MLP (Feedforward)'}")

    # Training durchführen
    history = train_trading_agent(
        env,
        agent,
        episodes=args.episodes,
        eval_every=args.eval_every,
        model_path=model_path
    )

    # Visualisiere Trainingsergebnisse
    results_path = config.get("visualization", "save_path", "results")
    os.makedirs(results_path, exist_ok=True)

    visualize_training(
        history,
        symbol,
        save_path=os.path.join(results_path, f"{symbol.replace('/', '_')}_training.png")
    )

    print("\n=== Training abgeschlossen ===")
    print(f"Bestes Modell gespeichert unter: {model_path}")


if __name__ == "__main__":
    main()