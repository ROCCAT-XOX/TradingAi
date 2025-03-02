import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from config.config import config


class AlpacaAPI:
    """Wrapper-Klasse für die Alpaca API."""

    def __init__(self, api_key=None, api_secret=None, base_url=None):
        """
        Initialisiert die Verbindung zur Alpaca API.

        Args:
            api_key: Alpaca API Key (optional, wird aus Config geladen)
            api_secret: Alpaca API Secret (optional, wird aus Config geladen)
            base_url: API URL (optional, wird aus Config geladen)
        """
        # Wenn keine Credentials übergeben wurden, aus Config laden
        if api_key is None or api_secret is None or base_url is None:
            api_key, api_secret, base_url = config.get_alpaca_credentials()

        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.account = self.api.get_account()
        print(f"API Verbindung hergestellt. Account Status: {self.account.status}")

    def get_historical_data(self, symbol=None, timeframe=None, start_date=None, end_date=None, limit=1000):
        """
        Holt historische Daten von Alpaca.

        Args:
            symbol: Aktien- oder Krypto-Symbol (z.B. 'AAPL' oder 'BTC/USD')
            timeframe: Zeitintervall ('1Min', '5Min', '15Min', '1H', '1D', etc.)
            start_date: Startdatum (Standard: 1000 Einheiten vor Ende)
            end_date: Enddatum (Standard: jetzt)
            limit: Maximale Anzahl an Datenpunkten

        Returns:
            DataFrame mit OHLCV-Daten
        """
        # Wenn Symbol nicht übergeben wurde, aus Config laden
        if symbol is None:
            symbol = config.get("trading", "symbol", "BTC/USD")

        if timeframe is None:
            timeframe = config.get("trading", "timeframe", "1D")

        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            lookback_days = config.get("backtesting", "lookback_days", 365)
            if timeframe == '1D':
                start_date = end_date - timedelta(days=lookback_days)
            elif timeframe == '1H':
                start_date = end_date - timedelta(hours=lookback_days)
            else:
                start_date = end_date - timedelta(minutes=lookback_days)

        # Format für Alpaca API - korrektes RFC3339-Format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Krypto oder Aktie bestimmen
        try:
            if '/' in symbol:  # Krypto
                bars = self.api.get_crypto_bars(symbol, timeframe, start=start_str, end=end_str).df
            else:  # Aktie
                bars = self.api.get_bars(symbol, timeframe, start=start_str, end=end_str).df

            # Aufräumen des DataFrames
            if bars.index.nlevels > 1:
                bars = bars.reset_index(level=0, drop=True)

            print(f"Historische Daten für {symbol} geladen: {len(bars)} Einträge")
            return bars

        except Exception as e:
            print(f"Fehler beim Laden der historischen Daten: {e}")
            return pd.DataFrame()

    def place_order(self, symbol, qty, side, order_type='market', time_in_force='gtc'):
        """
        Platziert eine Order.

        Args:
            symbol: Aktien- oder Krypto-Symbol
            qty: Anzahl der Einheiten
            side: 'buy' oder 'sell'
            order_type: 'market', 'limit', etc.
            time_in_force: 'day', 'gtc', etc.

        Returns:
            Order-Objekt oder None bei Fehler
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            print(f"Order platziert: {side} {qty} {symbol}")
            return order
        except Exception as e:
            print(f"Order-Fehler: {e}")
            return None

    def get_account_info(self):
        """
        Holt aktuelle Account-Informationen.

        Returns:
            dict: Account-Informationen
        """
        try:
            account = self.api.get_account()
            info = {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'regt_buying_power': float(account.regt_buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'status': account.status
            }
            return info
        except Exception as e:
            print(f"Fehler beim Abrufen der Account-Informationen: {e}")
            return {}

    def get_positions(self):
        """
        Holt alle aktuellen Positionen.

        Returns:
            dict: Positionen nach Symbol
        """
        try:
            positions = self.api.list_positions()
            result = {}
            for position in positions:
                result[position.symbol] = {
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                }
            return result
        except Exception as e:
            print(f"Fehler beim Abrufen der Positionen: {e}")
            return {}

    def get_latest_price(self, symbol):
        """
        Holt den aktuellen Preis eines Symbols.

        Args:
            symbol: Aktien- oder Krypto-Symbol

        Returns:
            float: Aktueller Preis oder None bei Fehler
        """
        try:
            if '/' in symbol:  # Krypto
                trades = self.api.get_crypto_trades(symbol, limit=1).df
                return trades.price.iloc[0]
            else:  # Aktie
                trades = self.api.get_latest_trade(symbol)
                return float(trades.price)
        except Exception as e:
            print(f"Fehler beim Abrufen des aktuellen Preises für {symbol}: {e}")
            return None