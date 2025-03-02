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
    Visualisiert die Backtesting-Ergebnisse.

    Args:
        df: DataFrame mit OHLCV-Daten
        results: Backtesting-Ergebnisse
        symbol: Gehandeltes Symbol
        save_path: Pfad zum Speichern der Visualisierung
    """
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
    initial_shares = results['initial_balance'] / initial_price
    buy_hold_values = [initial_shares * price for price in df['close'].values]
    ax1.plot(dates, buy_hold_values, 'b--', alpha=0.5, label='Buy & Hold')

    # Zweite y-Achse für Portfolio-Wert
    ax2 = ax1.twinx()

    # Länge anpassen (falls Portfolio-Werte kürzer sind)
    portfolio_values = results['portfolio_values']
    portfolio_dates = dates[-len(portfolio_values):]

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

    # Positionen plotten
    positions_dates = dates[-len(results['positions']):]
    plt.plot(positions_dates, results['positions'], 'b-', label='Position Size')
    plt.axhline(y=0, color='r', linestyle='--')

    # Formatierung
    plt.title('Position Size', fontsize=14)
    plt.ylabel('Units', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Aktionen (Buy, Hold, Sell)
    plt.subplot(3, 1, 3)

    # Preisdaten und Handelsentscheidungen
    prices = results['prices']
    actions = results['actions']
    action_dates = dates[-len(actions):]

    # Preislinie
    plt.plot(action_dates, prices, color='blue', alpha=0.5, label='Price')

    # Kaufsignale
    buys = [i for i, a in enumerate(actions) if a == 1]
    buy_dates = [action_dates[i] for i in buys]
    buy_prices = [prices[i] for i in buys]

    # Verkaufssignale
    sells = [i for i, a in enumerate(actions) if a == 2]
    sell_dates = [action_dates[i] for i in sells]
    sell_prices = [prices[i] for i in sells]

    # Signale plotten
    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')

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
        f'Backtesting Results for {symbol}\nReturn: {results["return"]:.2f}% | Buy & Hold: {results["buy_hold_return"]:.2f}%',
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

    # Plot 1: Kumulativer Return vs Buy & Hold
    plt.subplot(2, 2, 1)

    # Berechne kumulative Returns
    portfolio_returns = [v / results['initial_balance'] for v in results['portfolio_values']]

    # Buy & Hold normalisieren
    buy_hold_norm = [v / buy_hold_values[0] for v in buy_hold_values[-len(portfolio_returns):]]

    plt.plot(portfolio_dates, [r * 100 - 100 for r in portfolio_returns], 'g-', label='Strategy')
    plt.plot(portfolio_dates, [r * 100 - 100 for r in buy_hold_norm], 'b--', label='Buy & Hold')

    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Cumulative Return (%)', fontsize=14)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Drawdown
    plt.subplot(2, 2, 2)

    # Berechne Drawdowns
    portfolio_values = results['portfolio_values']
    drawdowns = []
    peak = portfolio_values[0]

    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        drawdowns.append(drawdown)

    plt.plot(portfolio_dates, drawdowns, 'r-')
    plt.axhline(y=results['max_drawdown'], color='k', linestyle='--',
                label=f'Max Drawdown: {results["max_drawdown"]:.2f}%')

    plt.title('Drawdown', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Balance vs. Position Value
    plt.subplot(2, 2, 3)

    balance_history = results['balance_history']
    position_values = [results['positions'][i] * results['prices'][i] for i in range(len(results['positions']))]

    plt.stackplot(portfolio_dates, [balance_history, position_values],
                  labels=['Cash', 'Position Value'],
                  colors=['#66c2a5', '#fc8d62'], alpha=0.7)

    plt.title('Portfolio Composition', fontsize=14)
    plt.ylabel('Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 4: Gewinnende vs. verlierende Trades
    plt.subplot(2, 2, 4)

    # Analyse der Trades
    if 'trades' in results and results['trades']:
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

    # Gesamttitel
    plt.suptitle(
        f'Performance Metrics for {symbol}\nSharpe Ratio: {results["sharpe_ratio"]:.2f} | Max Drawdown: {results["max_drawdown"]:.2f}%',
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