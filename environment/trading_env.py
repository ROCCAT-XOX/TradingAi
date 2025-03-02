import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """Trading Environment für Reinforcement Learning."""

    def __init__(self, df, initial_balance=10000, commission=0.001, window_size=10, render_mode=None):
        """
        Initialisiert die Trading-Umgebung.

        Args:
            df: DataFrame mit OHLCV-Daten und Indikatoren
            initial_balance: Anfangskapital
            commission: Handelsgebühr (z.B. 0.001 für 0.1%)
            window_size: Größe des Beobachtungsfensters für die Preishistorie
            render_mode: Visualisierungsmodus
        """
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.render_mode = render_mode

        # Aktionen: 0=Halten, 1=Kaufen, 2=Verkaufen
        self.action_space = spaces.Discrete(3)

        # Bestimme Dimensionen der Beobachtung
        self._get_observation_dimension()

        # Observation Space: [price_features, balance, position]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Initialisiere Zustand
        self.reset()

    def _get_observation_dimension(self):
        """Bestimmt die Dimension des Beobachtungsraums."""
        # Preisinformationen (OHLC) über window_size Zeitschritte
        price_dim = 4 * self.window_size  # Open, High, Low, Close

        # Indikatoren
        indicator_count = 0
        for col in self.df.columns:
            if col in ['sma_5', 'sma_10', 'sma_20', 'rsi_14', 'volatility_5d',
                       'upper_band', 'lower_band', 'volume_norm']:
                indicator_count += 1

        # Gesamtdimension: Preisinformationen + Indikatoren + Balance + Position
        self.obs_dim = price_dim + indicator_count + 2

    def _get_observation(self):
        """
        Erstellt die Beobachtung für den Agenten.

        Returns:
            numpy.ndarray: Beobachtungsvektor
        """
        # Indizes für das aktuelle Fenster
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        # Preisinformationen
        prices_window = self.df.iloc[start_idx:end_idx]

        # OHLC-Preise normalisieren (prozentuale Änderung zur ersten Beobachtung)
        first_close = prices_window['close'].iloc[0]

        price_obs = []
        for col in ['open', 'high', 'low', 'close']:
            if col in prices_window.columns:
                # Normalisiere auf prozentuale Änderung
                price_obs.extend(prices_window[col].values / first_close - 1)
            else:
                # Fallback: Spalte nicht vorhanden, fülle mit 0
                price_obs.extend([0] * self.window_size)

        # Indikatoren für den aktuellen Zeitschritt
        indicator_obs = []
        for col in ['sma_5', 'sma_10', 'sma_20', 'rsi_14', 'volatility_5d',
                    'upper_band', 'lower_band', 'volume_norm']:
            if col in self.df.columns:
                # Normalisiere auch den Indikator
                if col.startswith('sma_') or col in ['upper_band', 'lower_band']:
                    indicator_obs.append(self.df.iloc[end_idx - 1][col] / first_close - 1)
                elif col == 'rsi_14':
                    indicator_obs.append(self.df.iloc[end_idx - 1][col] / 100)  # RSI ist 0-100, auf 0-1 normalisieren
                else:
                    indicator_obs.append(self.df.iloc[end_idx - 1][col])  # Andere bereits normalisierte Werte

        # Account-Zustand
        balance_normalized = self.balance / self.initial_balance - 1  # Prozentuale Änderung
        position_normalized = self.shares * self.current_price() / self.initial_balance  # Position als Anteil am Startkapital

        # Gesamte Beobachtung zusammensetzen
        obs = np.array(price_obs + indicator_obs + [balance_normalized, position_normalized], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        """
        Setzt die Umgebung zurück.

        Args:
            seed: Zufallsseed
            options: Weitere Optionen

        Returns:
            tuple: (Beobachtung, Info-Dictionary)
        """
        super().reset(seed=seed)

        # Initialisiere Zustand
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.window_size
        self.done = False
        self.total_reward = 0
        self.portfolio_values = [self.initial_balance]
        self.trades = []

        return self._get_observation(), {}

    def step(self, action):
        """
        Führt eine Aktion aus und gibt den neuen Zustand zurück.

        Args:
            action: Aktion (0=Halten, 1=Kaufen, 2=Verkaufen)

        Returns:
            tuple: (Beobachtung, Belohnung, Beendet-Flag, Abgebrochen-Flag, Info-Dictionary)
        """
        # Aktuellen Portfolio-Wert vor der Aktion berechnen
        prev_portfolio_value = self.balance + self.shares * self.current_price()

        # Aktion ausführen
        self._take_action(action)

        # Zum nächsten Zeitschritt gehen
        self.current_step += 1

        # Prüfen, ob Episode beendet ist
        if self.current_step >= len(self.df):
            self.done = True

            # Alle verbleibenden Anteile verkaufen
            if self.shares > 0:
                sell_price = self.current_price() * (1 - self.commission)
                self.balance += self.shares * sell_price

                # Trade aufzeichnen
                self.trades.append({
                    'step': self.current_step - 1,
                    'type': 'sell',
                    'price': sell_price,
                    'shares': self.shares,
                    'value': self.shares * sell_price
                })

                self.shares = 0

        # Aktuellen Portfolio-Wert berechnen
        current_portfolio_value = self.balance + self.shares * self.current_price()
        self.portfolio_values.append(current_portfolio_value)

        # Belohnung berechnen
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        self.total_reward += reward

        # Info-Dictionary erstellen
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'current_price': self.current_price(),
            'total_reward': self.total_reward,
            'step': self.current_step
        }

        return self._get_observation(), reward, self.done, False, info

    def _take_action(self, action):
        """
        Führt die angegebene Aktion aus.

        Args:
            action: Aktion (0=Halten, 1=Kaufen, 2=Verkaufen)
        """
        current_price = self.current_price()

        if action == 1:  # Kaufen
            # Kaufe nur einen Teil des verfügbaren Kapitals (20%)
            max_shares = self.balance // (current_price * (1 + self.commission))
            buy_shares = max(1, int(max_shares * 0.2))  # Mindestens 1 Anteil, maximal 20% des möglichen

            if buy_shares > 0:
                self.shares += buy_shares
                cost = buy_shares * current_price * (1 + self.commission)
                self.balance -= cost

                # Trade aufzeichnen
                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': current_price,
                    'shares': buy_shares,
                    'value': cost
                })

        elif action == 2:  # Verkaufen
            if self.shares > 0:
                # Verkaufe 50% der Anteile statt alle
                sell_shares = max(1, int(self.shares * 0.5))
                sell_price = current_price * (1 - self.commission)
                gain = sell_shares * sell_price
                self.balance += gain

                # Trade aufzeichnen
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': sell_price,
                    'shares': sell_shares,
                    'value': gain
                })

                self.shares -= sell_shares

    def _calculate_reward(self, prev_value, current_value):
        """
        Berechnet die Belohnung für den aktuellen Schritt.

        Args:
            prev_value: Vorheriger Portfolio-Wert
            current_value: Aktueller Portfolio-Wert

        Returns:
            float: Belohnung
        """
        # Grundlegende prozentuale Änderung
        pct_change = (current_value / prev_value) - 1

        # Skaliere die Belohnung, um sie bedeutsamer zu machen
        scaled_reward = pct_change * 100  # Multiplikation mit 100 macht aus 0.01 → 1.0

        # Füge Handelsanreize hinzu
        if self.trades and self.trades[-1]['step'] == self.current_step:
            last_trade = self.trades[-1]
            if last_trade['type'] == 'buy':
                # Kleiner Abzug für Käufe, um unnötiges Handeln zu vermeiden
                scaled_reward -= 0.1
            elif last_trade['type'] == 'sell' and pct_change > 0:
                # Bonus für gewinnbringende Verkäufe
                scaled_reward += 0.5

        return scaled_reward

    def current_price(self):
        """
        Gibt den aktuellen Preis zurück.

        Returns:
            float: Aktueller Preis
        """
        return self.df['close'].iloc[min(self.current_step, len(self.df) - 1)]

    def render(self):
        """
        Visualisiert den aktuellen Zustand.
        """
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Price: ${self.current_price():.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares: {self.shares}")
            print(f"Portfolio Value: ${self.portfolio_values[-1]:.2f}")
            print(f"Total Reward: {self.total_reward:.4f}")
            print("-" * 50)

    def close(self):
        """
        Schließt die Umgebung.
        """
        pass