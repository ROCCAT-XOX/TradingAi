import pandas as pd
import numpy as np
from utils.alpaca_api import AlpacaAPI
from config.config import config


class DataLoader:
    """Klasse zum Laden und Vorverarbeiten von Handelsdaten."""

    def __init__(self, api=None):
        """
        Initialisiert den DataLoader.

        Args:
            api: AlpacaAPI-Instanz (optional)
        """
        self.api = api or AlpacaAPI()

    def load_historical_data(self, symbol=None, timeframe=None, start_date=None, end_date=None):
        """
        Lädt historische Daten und bereitet sie vor.

        Args:
            symbol: Aktien- oder Krypto-Symbol
            timeframe: Zeitintervall
            start_date: Startdatum
            end_date: Enddatum

        Returns:
            DataFrame mit vorverarbeiteten Daten
        """
        # Daten von Alpaca holen
        df = self.api.get_historical_data(symbol, timeframe, start_date, end_date)

        if df.empty:
            return df

        # Vorverarbeitung
        df = self._preprocess_data(df)

        return df

    def _preprocess_data(self, df):
        """
        Verarbeitet die Rohdaten vor.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            Vorverarbeiteter DataFrame
        """
        # Stelle sicher, dass die notwendigen Spalten vorhanden sind
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"WARNUNG: Spalte '{col}' fehlt im DataFrame")

        # Fehlende Werte behandeln
        df = df.ffill()  # Vorherigen Wert verwenden

        # Füge technische Indikatoren hinzu
        df = self._add_technical_indicators(df)

        # Normalisiere Volumen
        if 'volume' in df.columns:
            df['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()

        # Nur Zeilen ohne NaN-Werte behalten
        df = df.dropna()

        return df

    def _add_technical_indicators(self, df):
        """
        Fügt technische Indikatoren zum DataFrame hinzu.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit Indikatoren
        """
        # Kopie erstellen, um Warnungen zu vermeiden
        df = df.copy()

        # Gleitende Durchschnitte
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()

        # Relative Preisänderungen
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)

        # Volatilität
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()

        # Momentum-Indikatoren
        df['rsi_14'] = self._calculate_rsi(df, 14)

        # Preiskanäle
        df['upper_band'], df['lower_band'] = self._calculate_bollinger_bands(df, 20, 2)

        return df

    def _calculate_rsi(self, df, window=14):
        """
        Berechnet den Relative Strength Index (RSI).

        Args:
            df: DataFrame mit OHLCV-Daten
            window: Zeitfenster für die Berechnung

        Returns:
            Series mit RSI-Werten
        """
        # Kopie erstellen und mit Indizes arbeiten anstatt Zeitstempel
        price_df = df['close'].reset_index(drop=True)
        delta = price_df.diff()

        # Positive und negative Preisänderungen
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Exponentieller gleitender Durchschnitt
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        # Konvertiere zu Liste für die direkte Bearbeitung
        avg_gain_list = avg_gain.tolist()
        avg_loss_list = avg_loss.tolist()

        # Ersten gültigen Wert finden
        first_valid_idx = window

        # RSI nach ersten Werten berechnen
        for i in range(first_valid_idx, len(price_df)):
            avg_gain_list[i] = (avg_gain_list[i - 1] * (window - 1) + gain.iloc[i]) / window
            avg_loss_list[i] = (avg_loss_list[i - 1] * (window - 1) + loss.iloc[i]) / window

        # Zurück zu Series konvertieren
        avg_gain = pd.Series(avg_gain_list)
        avg_loss = pd.Series(avg_loss_list)

        # RS und RSI berechnen
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Zurück zum ursprünglichen Index
        rsi = pd.Series(rsi.values, index=df.index)

        return rsi

    def _calculate_bollinger_bands(self, df, window=20, num_std=2):
        """
        Berechnet Bollinger-Bänder.

        Args:
            df: DataFrame mit OHLCV-Daten
            window: Zeitfenster für gleitenden Durchschnitt
            num_std: Anzahl der Standardabweichungen

        Returns:
            tuple: (Oberes Band, Unteres Band)
        """
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return upper_band, lower_band

    def split_train_test(self, df, test_size=0.2):
        """
        Teilt Daten in Trainings- und Testdaten.

        Args:
            df: DataFrame mit vorverarbeiteten Daten
            test_size: Anteil der Testdaten

        Returns:
            tuple: (Trainings-DataFrame, Test-DataFrame)
        """
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        return train_df, test_df