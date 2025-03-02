import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pathlib import Path


def visualize_training(history, symbol, save_path=None):
    """
    Visualisiert den Trainingsverlauf.

    Args:
        history: Trainingsverlauf
        symbol: Gehandeltes Symbol
        save_path: Pfad zum Speichern der Visualisierung
    """
    # Seaborn-Stil festlegen
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 12))

    # Plot 1: Episode Rewards
    plt.subplot(2, 2, 1)
    plt.plot(history['episode_rewards'], 'b-')
    plt.title('Episode Rewards', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Gleitender Durchschnitt hinzufügen
    if len(history['episode_rewards']) > 5:
        window = min(10, len(history['episode_rewards']) // 3)
        rolling_mean = pd.Series(history['episode_rewards']).rolling(window=window).mean()
        plt.plot(rolling_mean, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
        plt.legend()

    # Plot 2: Final Portfolio Values
    plt.subplot(2, 2, 2)
    final_values = [values[-1] for values in history['portfolio_values']]
    plt.plot(final_values, 'g-')
    plt.axhline(y=history['portfolio_values'][0][0], color='r', linestyle='--', label='Initial Value')
    plt.title('Final Portfolio Values', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Evaluation Returns
    plt.subplot(2, 2, 3)
    if history['eval_returns']:
        eval_episodes = [(i + 1) * 10 for i in range(len(history['eval_returns']))]
        plt.plot(eval_episodes, history['eval_returns'], 'b-', marker='o')
        plt.title('Evaluation Returns', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No evaluation data available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)

    # Plot 4: Best Episode Portfolio Growth
    plt.subplot(2, 2, 4)
    best_episode = np.argmax([values[-1] for values in history['portfolio_values']])
    best_portfolio_values = history['portfolio_values'][best_episode]

    plt.plot(best_portfolio_values, 'g-')
    plt.title(f'Best Episode Portfolio Growth - Episode {best_episode + 1}', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Format y-axis as currency
    def currency_formatter(x, pos):
        return f'${x:,.0f}'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Gesamttitel
    plt.suptitle(f'Training Results for {symbol}', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Speichern oder Anzeigen
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trainingsergebnisse gespeichert unter: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_backtest(df, results, symbol, save_path=None):
    """
    Visualisiert die Backtesting-Ergebnisse mit robuster Fehlerbehandlung.

    Args:
        df: DataFrame mit OHLCV-Daten
        results: Backtesting-Ergebnisse
        symbol: Gehandeltes Symbol
        save_path: Pfad zum Speichern der Visualisierung
    """
    # Robust fehlerbehandlung: Überprüfe ob die erforderlichen Schlüssel existieren
    required_keys = ['positions', 'prices', 'portfolio_values', 'balance_history',
                     'actions', 'return', 'buy_hold_return', 'max_drawdown', 'sharpe_ratio']

    # Prüfe alle erforderlichen Schlüssel
    for key in required_keys:
        if key not in results:
            print(f"Warnung: '{key}' fehlt in den Backtest-Ergebnissen")
            results[key] = []

        # Wenn der Schlüssel existiert, aber leer oder None ist
        if not results[key] and isinstance(results[key], (list, dict, np.ndarray)):
            print(f"Warnung: '{key}' ist leer in den Backtest-Ergebnissen")
            if key in ['return', 'buy_hold_return', 'max_drawdown', 'sharpe_ratio']:
                results[key] = 0.0
            else:
                results[key] = []

    # Seaborn-Stil festlegen
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 14))

    # Zeitachse erstellen, falls Daten einen Zeitindex haben
    has_date_index = False

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
        has_date_index = True
    else:
        dates = range(len(df))

    # Plot 1: Preise und Portfolio-Wert
    ax1 = plt.subplot(3, 1, 1)

    # Preise plotten
    ax1.plot(dates, df['close'].values, color='blue', label='Price', alpha=0.7)
    ax1.set_ylabel('Price', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Buy & Hold Linie hinzufügen
    initial_price = df['close'].iloc[0]
    initial_shares = results.get('initial_balance', 10000) / initial_price
    buy_hold_values = [initial_shares * price for price in df['close'].values]
    ax1.plot(dates, buy_hold_values, 'b--', alpha=0.5, label='Buy & Hold')

    # Zweite y-Achse für Portfolio-Wert
    ax2 = ax1.twinx()

    # Länge anpassen (falls Portfolio-Werte kürzer sind)
    portfolio_values = results.get('portfolio_values', [results.get('initial_balance', 10000)])
    portfolio_dates = dates[-len(portfolio_values):] if len(portfolio_values) <= len(dates) else dates

    ax2.plot(portfolio_dates, portfolio_values, color='green', label='Portfolio Value')
    ax2.set_ylabel('Portfolio Value ($)', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')

    # Format y-axis as currency
    def currency_formatter(x, pos):
        return f'${x:,.0f}'

    ax2.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Titel und Legende
    plt.title(f'{symbol} Price and Portfolio Value', fontsize=14)

    # Kombinierte Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot 2: Positionen
    plt.subplot(3, 1, 2)

    # Positionen plotten, mit Fehlerbehandlung
    positions = results.get('positions', [])
    if len(positions) > 0:
        positions_dates = dates[-len(positions):] if len(positions) <= len(dates) else dates
        plt.plot(positions_dates, positions, 'b-', label='Position Size')
        plt.axhline(y=0, color='r', linestyle='--')
    else:
        plt.text(0.5, 0.5, 'No position data available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)

    # Formatierung
    plt.title('Position Size', fontsize=14)
    plt.ylabel('Units', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Aktionen (Buy, Hold, Sell)
    plt.subplot(3, 1, 3)

    # Preisdaten und Handelsentscheidungen
    prices = results.get('prices', [])
    actions = results.get('actions', [])

    # Robuste Handhabung der Daten
    if len(prices) > 0 and len(actions) > 0:
        # Sichere Berechnung der Aktionsdaten
        min_length = min(len(actions), len(dates))
        action_dates = dates[-min_length:] if min_length <= len(dates) else dates

        # Minimallänge für Preise finden
        min_price_length = min(len(prices), min_length)
        plot_prices = prices[:min_price_length]
        plot_action_dates = action_dates[:min_price_length]

        # Preislinie
        plt.plot(plot_action_dates, plot_prices, color='blue', alpha=0.5, label='Price')

        # Verarbeitung der Aktionsdaten
        if len(actions) > 0 and len(prices) > 0:
            # Kaufsignale - mit Fehlerbehandlung
            try:
                buys = [i for i, a in enumerate(actions) if a == 1 and i < len(prices)]
                buy_dates = [action_dates[i] for i in buys if i < len(action_dates)]
                buy_prices = [prices[i] for i in buys if i < len(prices)]

                # Verkaufssignale - mit Fehlerbehandlung
                sells = [i for i, a in enumerate(actions) if a == 2 and i < len(prices)]
                sell_dates = [action_dates[i] for i in sells if i < len(action_dates)]
                sell_prices = [prices[i] for i in sells if i < len(prices)]

                # Signale plotten
                if buy_dates and buy_prices:
                    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
                if sell_dates and sell_prices:
                    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')
            except Exception as e:
                print(f"Fehler beim Plotten der Handelssignale: {e}")
    else:
        plt.text(0.5, 0.5, 'No trading action data available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)

    # Formatierung
    plt.title('Trading Actions', fontsize=14)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Formatiere x-Achse als Datum, falls verfügbar
    if has_date_index:
        for ax in plt.gcf().axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Gesamttitel
    plt.suptitle(
        f'Backtesting Results for {symbol}\nReturn: {results.get("return", 0):.2f}% | Buy & Hold: {results.get("buy_hold_return", 0):.2f}%',
        fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Speichern oder Anzeigen
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Backtesting-Ergebnisse gespeichert unter: {save_path}")
    else:
        plt.show()

    plt.close()

    # Zweite Abbildung: Performance-Metriken
    plt.figure(figsize=(12, 10))

    # Robuste Fehlerbehandlung für die zweite Abbildung
    if (len(results.get('portfolio_values', [])) > 1 and
            len(buy_hold_values) > 1 and
            len(portfolio_dates) > 1):

        # Plot 1: Kumulativer Return vs Buy & Hold
        plt.subplot(2, 2, 1)

        # Berechne kumulative Returns mit Fehlerbehandlung
        try:
            portfolio_returns = [v / results.get('initial_balance', 10000) for v in results.get('portfolio_values', [])]

            # Buy & Hold normalisieren
            buy_hold_len = min(len(buy_hold_values), len(portfolio_dates))
            buy_hold_norm = [v / buy_hold_values[0] for v in buy_hold_values[-buy_hold_len:]]

            plt.plot(portfolio_dates, [r * 100 - 100 for r in portfolio_returns], 'g-', label='Strategy')

            plot_dates = portfolio_dates[:len(buy_hold_norm)] if len(portfolio_dates) > len(
                buy_hold_norm) else portfolio_dates
            plt.plot(plot_dates, [r * 100 - 100 for r in buy_hold_norm[:len(plot_dates)]], 'b--', label='Buy & Hold')

            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Cumulative Return (%)', fontsize=14)
            plt.ylabel('Return (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
        except Exception as e:
            print(f"Fehler beim Plotten der kumulativen Returns: {e}")
            plt.text(0.5, 0.5, 'Error calculating returns',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)

        # Plot 2: Drawdown
        plt.subplot(2, 2, 2)

        # Berechne Drawdowns mit Fehlerbehandlung
        try:
            portfolio_values = results.get('portfolio_values', [results.get('initial_balance', 10000)])
            drawdowns = []
            peak = portfolio_values[0]

            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                drawdowns.append(drawdown)

            plt.plot(portfolio_dates, drawdowns, 'r-')
            plt.axhline(y=results.get('max_drawdown', 0), color='k', linestyle='--',
                        label=f'Max Drawdown: {results.get("max_drawdown", 0):.2f}%')

            plt.title('Drawdown', fontsize=14)
            plt.ylabel('Drawdown (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
        except Exception as e:
            print(f"Fehler beim Plotten der Drawdowns: {e}")
            plt.text(0.5, 0.5, 'Error calculating drawdowns',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)

        # Plot 3: Balance vs. Position Value
        plt.subplot(2, 2, 3)

        try:
            balance_history = results.get('balance_history', [results.get('initial_balance', 10000)])
            positions = results.get('positions', [0] * len(balance_history))
            prices = results.get('prices', [0] * len(positions))

            # Stellen Sie sicher, dass alle Listen gleich lang sind
            min_length = min(len(balance_history), len(positions), len(prices))

            # Wenn wir genug Daten haben, erstellen wir den Stackplot
            if min_length > 1:
                # Vorsichtig die Position-Werte berechnen
                position_values = []
                for i in range(min_length):
                    position_values.append(positions[i] * prices[i] if i < len(prices) else 0)

                # Daten für den Stackplot vorbereiten
                plot_dates = portfolio_dates[:min_length]
                plot_balance = balance_history[:min_length]
                plot_positions = position_values[:min_length]

                plt.stackplot(plot_dates, [plot_balance, plot_positions],
                              labels=['Cash', 'Position Value'],
                              colors=['#66c2a5', '#fc8d62'], alpha=0.7)

                plt.title('Portfolio Composition', fontsize=14)
                plt.ylabel('Value ($)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'Insufficient data for portfolio composition',
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes, fontsize=14)
        except Exception as e:
            print(f"Fehler beim Plotten der Portfolio-Zusammensetzung: {e}")
            plt.text(0.5, 0.5, 'Error calculating portfolio composition',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)

        # Plot 4: Gewinnende vs. verlierende Trades
        plt.subplot(2, 2, 4)

        # Analyse der Trades
        if 'trades' in results and results['trades']:
            try:
                trades = results['trades']
                buy_trades = [t for t in trades if t['type'] == 'buy']
                sell_trades = [t for t in trades if t['type'] == 'sell']

                # Berechne Gewinn/Verlust pro Trade
                trade_profits = []
                current_trade = None

                for trade in trades:
                    if trade['type'] == 'buy':
                        current_trade = trade
                    elif trade['type'] == 'sell' and current_trade is not None:
                        # Berechne Gewinn/Verlust
                        buy_value = current_trade['value']
                        sell_value = trade['value']
                        profit = sell_value - buy_value
                        trade_profits.append(profit)
                        current_trade = None

                # Gewinnende und verlierende Trades zählen
                winning_trades = sum(1 for p in trade_profits if p > 0)
                losing_trades = sum(1 for p in trade_profits if p <= 0)

                # Kreisdiagramm erstellen
                labels = ['Gewinnend', 'Verlierend']
                sizes = [winning_trades, losing_trades]
                colors = ['#66c2a5', '#fc8d62']

                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        startangle=90, wedgeprops={'alpha': 0.7})

                plt.title('Trade Performance', fontsize=14)

                # Zusätzliche Informationen
                win_rate = winning_trades / len(trade_profits) * 100 if trade_profits else 0
                plt.annotate(f'Win Rate: {win_rate:.1f}%\nTotal Trades: {len(trade_profits)}',
                             xy=(0.5, 0.02), xycoords='axes fraction',
                             horizontalalignment='center', verticalalignment='bottom')
            except Exception as e:
                print(f"Fehler beim Analysieren der Trades: {e}")
                plt.text(0.5, 0.5, 'Error analyzing trades',
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes, fontsize=14)
        else:
            plt.text(0.5, 0.5, 'No trade data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)

        # Formatiere x-Achse als Datum, falls verfügbar
        if has_date_index:
            for i, ax in enumerate(plt.gcf().axes):
                if i != 3:  # Überspringe das Kreisdiagramm
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        # Wenn nicht genügend Daten für die Diagramme vorhanden sind
        for i in range(1, 5):
            plt.subplot(2, 2, i)
            plt.text(0.5, 0.5, 'Insufficient data for analysis',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)

    # Gesamttitel
    plt.suptitle(
        f'Performance Metrics for {symbol}\nSharpe Ratio: {results.get("sharpe_ratio", 0):.2f} | Max Drawdown: {results.get("max_drawdown", 0):.2f}%',
        fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Speichern oder Anzeigen
    if save_path:
        # Erstelle einen zweiten Dateinamen für die Metriken
        metrics_path = Path(save_path)
        metrics_path = metrics_path.parent / f"{metrics_path.stem}_metrics{metrics_path.suffix}"
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"Performance-Metriken gespeichert unter: {metrics_path}")
    else:
        plt.show()

    plt.close()