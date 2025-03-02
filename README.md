# Trading AI mit Reinforcement Learning

Dieses Projekt implementiert einen Reinforcement Learning-Agenten für den automatisierten Handel mit Aktien und Kryptowährungen. Der Agent wird mit PyTorch implementiert und verwendet den Actor-Critic-Algorithmus, um optimale Handelsentscheidungen zu treffen.

## Funktionen

- **Alpaca API Integration**: Nahtlose Verbindung zur Alpaca Markets API für den Zugriff auf Marktdaten und das Ausführen von Trades
- **Reinforcement Learning Environment**: Angepasste Gymnasium-Umgebung für das Trading-Szenario
- **Actor-Critic Netzwerk**: Neuronales Netzwerk für die Entscheidungsfindung
- **Trainingsvisualisierung**: Umfassende Visualisierung des Trainingsprozesses und der Ergebnisse
- **Backtesting**: Bewertung der Strategie anhand historischer Daten
- **Konfigurierbarkeit**: Einfaches Anpassen der Parameter über Konfigurationsdateien

## Installation

1. Repository klonen:
   ```
   git clone https://github.com/yourusername/trading-ai-project.git
   cd trading-ai-project
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```
   python -m venv venv
   source venv/bin/activate  # Unter Windows: venv\Scripts\activate
   ```

3. Abhängigkeiten installieren:
   ```
   pip install -r requirements.txt
   ```

4. Konfiguration einrichten:
   - Bearbeite `config/settings.json` mit deinen API-Schlüsseln
   - Oder setze Umgebungsvariablen (empfohlen für mehr Sicherheit):
     ```
     export ALPACA_API_KEY=your_api_key
     export ALPACA_API_SECRET=your_api_secret
     ```

## Konfiguration

Die Haupteinstellungen können in der `config/settings.json` Datei angepasst werden:

- **API-Einstellungen**: Alpaca API-Schlüssel und Endpunkt
- **Trading-Parameter**: Symbol, Zeitrahmen, Startkapital, Gebühren
- **Modell-Parameter**: Fenstergröße, Netzwerkgröße, Lernrate, Discount-Faktor
- **Trainingseinstellungen**: Anzahl der Episoden, Evaluierungsintervall
- **Visualisierungseinstellungen**: Speicherpfade für Diagramme und Logs

## Verwendung

### Training

```
python train.py
```

Dies startet den Trainingsprozess mit den Einstellungen aus der Konfigurationsdatei. Das beste Modell wird automatisch gespeichert.

### Backtesting

```
python backtest.py
```

Führt eine Backtesting-Analyse mit dem besten Modell durch und erstellt Visualisierungen.

### Hauptprogramm

```
python main.py
```

Führt den kompletten Workflow aus: Datenerfassung, Training und Backtesting.

## Projektstruktur

```
trading_ai_project/
│
├── config/              # Konfigurationsdateien
│   ├── settings.json    # Hauptkonfiguration
│   └── config.py        # Konfigurationsklasse
│
├── data/                # Daten-Management
│   └── data_loader.py   # Laden und Vorverarbeiten von Daten
│
├── models/              # Modellarchitektur
│   ├── actor_critic.py  # Actor-Critic-Netzwerk
│   └── trading_agent.py # RL-Agent
│
├── environment/         # Trading-Umgebung
│   └── trading_env.py   # Gymnasium-Umgebung
│
├── utils/               # Hilfsfunktionen
│   ├── alpaca_api.py    # Alpaca API-Wrapper
│   └── visualizer.py    # Visualisierungsfunktionen
│
├── train.py             # Trainingsscript
├── backtest.py          # Backtesting-Script
├── main.py              # Hauptprogramm
└── requirements.txt     # Abhängigkeiten
```

## Erweiterungsmöglichkeiten

- **Mehrere Vermögenswerte**: Erweitern des Agenten für das gleichzeitige Handeln mehrerer Assets
- **Zusätzliche Funktionen**: Integration von technischen Indikatoren
- **Hyperparameter-Optimierung**: Automatische Suche nach optimalen Parametern
- **Live-Trading**: Implementierung eines Live-Trading-Modus

## Lizenz

MIT

## Mitwirkende

Füge hier deinen Namen hinzu!