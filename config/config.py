import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Lade .env Datei, falls vorhanden
load_dotenv()


class Config:
    """Konfigurationsklasse für das TradingAI-Projekt."""

    def __init__(self, config_path=None):
        """
        Initialisiert die Konfiguration.

        Args:
            config_path: Pfad zur settings.json Datei (optional)
        """
        # Finde den Projektordner
        self.project_dir = Path(__file__).parent.parent

        # Standard-Konfigurationspfad
        if config_path is None:
            config_path = self.project_dir / "config" / "settings.json"

        # Lade Konfiguration aus JSON
        try:
            with open(config_path, 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            print(f"WARNUNG: Konfigurationsdatei nicht gefunden: {config_path}")
            self.settings = {}

        # Überschreibe mit Umgebungsvariablen falls vorhanden
        self._override_from_env()

    def _override_from_env(self):
        """Überschreibt Einstellungen mit Umgebungsvariablen."""
        # API-Schlüssel
        if "ALPACA_API_KEY" in os.environ:
            self.settings.setdefault("api", {}).setdefault("alpaca", {})["api_key"] = os.environ["ALPACA_API_KEY"]

        if "ALPACA_API_SECRET" in os.environ:
            self.settings.setdefault("api", {}).setdefault("alpaca", {})["api_secret"] = os.environ["ALPACA_API_SECRET"]

        # Trading-Symbol
        if "TRADING_SYMBOL" in os.environ:
            self.settings.setdefault("trading", {})["symbol"] = os.environ["TRADING_SYMBOL"]

    def get(self, section, key=None, default=None):
        """
        Holt eine Einstellung aus der Konfiguration.

        Args:
            section: Abschnitt in der Konfiguration
            key: Schlüssel innerhalb des Abschnitts (optional)
            default: Standardwert, falls Einstellung nicht gefunden

        Returns:
            Einstellungswert oder Standardwert
        """
        if section not in self.settings:
            return default

        if key is None:
            return self.settings[section]

        return self.settings[section].get(key, default)

    def get_alpaca_credentials(self):
        """
        Holt die Alpaca API-Zugangsdaten.

        Returns:
            tuple: (api_key, api_secret, base_url)
        """
        alpaca = self.get("api", "alpaca", {})
        return (
            alpaca.get("api_key", ""),
            alpaca.get("api_secret", ""),
            alpaca.get("base_url", "https://paper-api.alpaca.markets")
        )

    def save(self, config_path=None):
        """
        Speichert die aktuelle Konfiguration.

        Args:
            config_path: Pfad zum Speichern (optional)
        """
        if config_path is None:
            config_path = self.project_dir / "config" / "settings.json"

        # Stelle sicher, dass der Ordner existiert
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(self.settings, f, indent=2)


# Globale Konfigurationsinstanz
config = Config()