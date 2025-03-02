import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config.config import config
from utils.alpaca_api import AlpacaAPI
from data.data_loader import DataLoader
from models.trading_agent import TradingAgent
from environment.trading_env import TradingEnvironment
from utils.visualizer import visualize_backtest


def parse_args():
    """Kommandozeilenargumente parsen."""
    parser = argparse.ArgumentParser(description="Trading AI Backtesting")

    parser.add_argument('--symbol', type=str,
                        help='Trading-Symbol (z.B. BTC/USD oder AAPL)')

    parser.add_argument('--model-path', type=str,
                        help='Pfad zum gespeicherten Modell')

    parser.add_argument('--window', type=int, default=10,
                        help='Größe des Beobachtungsfensters')

    parser.add_argument('--balance', type=float, default=10000,
                        help='Anfangskapital')

    parser.add_argument('--commission', type=float, default=0.001,
                        help='Handelsgebühr (z.B. 0.001 für 0.1%)')

    parser.add_argument('--use-lstm', action='store_true',
                        help='LSTM-Netzwerk verwenden')

    parser.add_argument('--test-split', type=float, default=0.3,
                        help='Anteil der Daten für Backtesting (0.0-1.0)')

    return parser.parse_args()


def run_backtest(df, agent, env_params, test_df=None):
    """
    Führt Backtesting mit dem trainierten Agenten durch.

    Args:
        df: DataFrame mit OHLCV-Daten
        agent: Trainierter Agent
        env_params: Parameter für die Trading-Umgebung
        test_df: Separater Test-DataFrame (optional, sonst wird df verwendet)

    Returns:
        dict: Backtesting-Ergebnisse
    """
    # Verwende separaten Test-DataFrame, falls angegeben
    test_data = test_df if test_df is not None else df

    # Neue Umgebung für Backtesting erstellen
    env = TradingEnvironment(test_data, **env_params)

    # Anfangszustand
    state, _ = env.reset()
    done = False

    # Speicher für Ergebnisse
    actions_taken = []
    portfolio_values = [env.initial_balance]
    positions = [0]
    prices = []
    balance_history = [env.initial_balance]

    # Backtest durchlaufen
    print("Starte Backtesting...")
    while not done:
        # Wähle beste Aktion (ohne Exploration)
        action = agent.select_action(state, training=False)

        # Führe Aktion aus
        next_state, reward, done, _, info = env.step(action)

        # Speichere Ergebnisse
        actions_taken.append(action)
        portfolio_values.append(info['portfolio_value'])
        positions.append(info['shares'])
        prices.append(info['current_price'])
        balance_history.append(info['balance'])

        state = next_state

    # Berechne Metriken
    initial_price = test_data['close'].iloc[env.window_size]
    final_price = test_data['close'].iloc[-1]

    price_return = (final_price / initial_price - 1) * 100
    strategy_return = (portfolio_values[-1] / env.initial_balance - 1) * 100

    # Berechne Drawdown
    max_drawdown = 0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Sharpe Ratio (vereinfachte Berechnung)
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = portfolio_values[i] / portfolio_values[i - 1] - 1
        daily_returns.append(daily_return)

    mean_return = np.mean(daily_returns) if daily_returns else 0
    std_return = np.std(daily_returns) if daily_returns else 1

    # Annualisierte Sharpe Ratio (angenommen: tägliche Daten)
    # Riskofreie Rendite wird vernachlässigt
    sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0

    # Analysiere Trades
    trades = env.trades
    profitable_trades = sum(1 for trade in trades if trade['type'] == 'sell' and trade['value'] > 0)
    total_trades = len([trade for trade in trades if trade['type'] == 'sell'])
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    # Ergebnisse zusammenstellen
    results = {
        'initial_balance': env.initial_balance,
        'final_balance': portfolio_values[-1],
        'return': strategy_return,
        'buy_hold_return': price_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'trades': trades,
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'portfolio_values': portfolio_values,
        'positions': positions,
        'actions': actions_taken,
        'prices': prices,
        'balance_history': balance_history
    }

    # Ergebnisse ausgeben
    print("\n=== Backtesting-Ergebnisse ===")
    print(f"Anfangskapital: ${results['initial_balance']:.2f}")
    print(f"Endkapital: ${results['final_balance']:.2f}")
    print(f"Strategie-Return: {results['return']:.2f}%")
    print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
    print(f"Überperformance: {results['return'] - results['buy_hold_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Anzahl Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")

    return results


def main():
    """Hauptfunktion für das Backtesting."""
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

    # Teile Daten in Trainings- und Testset
    train_df, test_df = data_loader.split_train_test(df, test_size=args.test_split)

    print(f"Daten geladen: {len(df)} Einträge gesamt")
    print(f"Trainingsset: {len(train_df)} Einträge, Testset: {len(test_df)} Einträge")

    # Trading Environment Parameter
    env_params = {
        'initial_balance': args.balance,
        'commission': args.commission,
        'window_size': args.window
    }

    # Agent initialisieren
    agent_params = {
        'input_dim': len(df.columns) * args.window + 2,  # Feature-Spalten * Fenster + Balance + Position
        'n_actions': 3,  # Halten, Kaufen, Verkaufen
        'use_lstm': args.use_lstm
    }

    agent = TradingAgent(**agent_params)

    # Modellpfad bestimmen
    if args.model_path:
        model_path = args.model_path
    else:
        # Standard-Modellpfad
        save_path = config.get("training", "save_path", "saved_models")
        model_filename = f"{symbol.replace('/', '_')}_{'lstm' if args.use_lstm else 'mlp'}_model.pth"
        model_path = os.path.join(save_path, model_filename)

    # Modell laden
    if not agent.load(model_path):
        print(f"Kein Modell gefunden unter {model_path}")
        print("Beende Programm.")
        return

    # Backtesting durchführen
    results = run_backtest(df, agent, env_params, test_df)

    # Visualisiere Backtesting-Ergebnisse
    results_path = config.get("visualization", "save_path", "results")
    os.makedirs(results_path, exist_ok=True)

    visualize_backtest(
        test_df,
        results,
        symbol,
        save_path=os.path.join(results_path, f"{symbol.replace('/', '_')}_backtest.png")
    )

    print(f"\nBacktesting abgeschlossen. Ergebnisse gespeichert unter: {results_path}")


if __name__ == "__main__":
    main()