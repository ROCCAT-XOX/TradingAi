# Trading AI mit Reinforcement Learning

Dieses Projekt implementiert einen Reinforcement Learning-Agenten für den automatisierten Handel mit Aktien und Kryptowährungen. Der Agent wird mit PyTorch implementiert und verwendet den Actor-Critic-Algorithmus, um optimale Handelsentscheidungen zu treffen.

## Funktionen

- **Alpaca API Integration**: Nahtlose Verbindung zur Alpaca Markets API für den Zugriff auf Marktdaten und das Ausführen von Trades
- **Reinforcement Learning Environment**: Angepasste Gymnasium-Umgebung für das Trading-Szenario
- **Actor-Critic Netzwerk**: Neuronales Netzwerk für die Entscheidungsfindung
- **LSTM-Unterstützung**: Optionales LSTM-Netzwerk für bessere temporale Verarbeitung
- **Trainingsvisualisierung**: Umfassende Visualisierung des Trainingsprozesses und der Ergebnisse
- **Backtesting**: Bewertung der Strategie anhand historischer Daten
- **Konfigurierbarkeit**: Einfaches Anpassen der Parameter über Konfigurationsdateien oder Kommandozeilenargumente
- **Multi-Asset Support**: Training auf verschiedenen Vermögenswerten (BTC, ETH, SOL, Aktien, etc.)
- **CUDA-Beschleunigung**: Optimierte Performance-Nutzung von NVIDIA GPUs für schnelleres Training

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

4. Für GPU-Beschleunigung, installiere PyTorch mit CUDA-Unterstützung:
   ```
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (Ersetze cu118 mit deiner CUDA-Version)

5. Konfiguration einrichten:
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
python train.py --symbol BTC/USD --episodes 300 --lr 0.0001 --use-lstm
```

Dies trainiert ein LSTM-Modell für Bitcoin mit 300 Episoden und einer Lernrate von 0.0001.

Weitere Optionen:
- `--gamma 0.99`: Discount-Faktor für zukünftige Belohnungen setzen
- `--window-size 20`: Größe des Beobachtungsfensters ändern
- `--hidden-dim 256`: Größe der versteckten Schichten im Netzwerk anpassen
- `--timeframe 1D`: Zeitintervall der Daten (1Min, 5Min, 15Min, 1H, 4H, 1D)
- `--lookback 365`: Anzahl der Zeiteinheiten aus der Vergangenheit

### Backtesting

```
python backtest.py --symbol BTC/USD --use-lstm
```

Führt eine Backtesting-Analyse mit dem besten Modell durch und erstellt Visualisierungen.

### Hauptprogramm

```
python main.py --symbol BTC/USD --episodes 300 --lr 0.0001 --use-lstm
```

Führt den kompletten Workflow aus: Datenerfassung, Training und Backtesting.

Beispiele für verschiedene Zeitrahmen:

```
# Training mit 5-Minuten-Daten der letzten 1000 Zeitpunkte
python main.py --symbol BTC/USD --timeframe 5Min --lookback 1000 --episodes 300 --use-lstm

# Training mit Stundendaten der letzten 720 Stunden (30 Tage)
python main.py --symbol ETH/USD --timeframe 1H --lookback 720 --episodes 200 --use-lstm

# Training mit Tagesdaten der letzten 90 Tage
python main.py --symbol AAPL --timeframe 1D --lookback 90 --episodes 150 --use-lstm
```

Optionen für `main.py`:
- `--mode train`: Nur Training ausführen
- `--mode backtest`: Nur Backtesting ausführen (erfordert ein vorhandenes Modell)
- `--mode all`: Training und Backtesting ausführen (Standard)

### Unterstützte Vermögenswerte

Das System funktioniert mit allen von Alpaca Markets unterstützten Assets:
- Kryptowährungen: BTC/USD, ETH/USD, SOL/USD, etc.
- Aktien: AAPL, MSFT, GOOGL, etc.

## Optimierte Handelsstrategien

Die aktuelle Implementierung verwendet folgende Handelsstrategien:

- **Teilweise Käufe**: Der Agent kauft 10% des verfügbaren Kapitals pro Kaufsignal
- **Teilweise Verkäufe**: Der Agent verkauft 30% der Anteile pro Verkaufssignal
- **Epsilon-Greedy Exploration**: 10% zufällige Aktionen während des Trainings für bessere Exploration
- **Belohnungsoptimierung**: Gezielte Anreize für gewinnbringende Verkäufe und moderater Abzug für Käufe

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
├── train.py             # Trainingsskript
├── backtest.py          # Backtesting-Skript
├── main.py              # Hauptprogramm
└── requirements.txt     # Abhängigkeiten
```

## Ergebnisauswertung

Nach dem Training und Backtesting werden folgende Metriken generiert:

- **Gesamtrendite**: Prozentuale Rendite der Strategie
- **Vergleich mit Buy & Hold**: Überperformance gegenüber einfacher Buy & Hold-Strategie
- **Sharpe Ratio**: Risikobereinigte Rendite
- **Drawdown**: Maximaler Rückgang des Portfoliowerts
- **Win Rate**: Anteil der gewinnbringenden Trades
- **Handelsvisualisierung**: Graphische Darstellung der Kauf- und Verkaufssignale

Die Ergebnisse werden im `results`-Verzeichnis als PNG-Dateien gespeichert.

## Tipps für bessere Performance

- **Längeres Training**: 300-500 Episoden für stabilere Ergebnisse
- **Niedrigere Lernrate**: 0.0001 statt 0.001 für stabilere Konvergenz
- **LSTM verwenden**: Bessere Erkennung von Zeitmustern in den Marktdaten
- **Fenstergröße anpassen**: Größere Fenster (15-30) für langfristigere Muster
- **Hyperparameter-Tuning**: Verschiedene Kombinationen von Gamma und Lernrate testen
- **GPU-Beschleunigung**: Verwende CUDA für erheblich schnelleres Training
- **Zeitrahmenanpassung**: Für kurzfristige Strategien Minuten-Daten, für langfristige Tages-Daten

## Fehlerbehebung

- **Dimension Error**: Bei Dimension-Fehlern sicherstellen, dass die Eingabedimension des Netzwerks mit der Observation-Dimension der Umgebung übereinstimmt
- **Extreme Schwankungen**: Bei extremen Schwankungen im Training die Lernrate reduzieren und das Gradient Clipping verstärken
- **API-Fehler**: Sicherstellen, dass die Alpaca API-Schlüssel korrekt konfiguriert sind und das gewünschte Symbol unterstützt wird
- **CUDA-Probleme**: Bei GPU-Fehlern sicherstellen, dass PyTorch mit der richtigen CUDA-Version installiert ist
- **Out of Memory**: Bei Speicherfehlern die Batch-Größe reduzieren oder auf eine GPU mit mehr VRAM wechseln

## Erweiterungsmöglichkeiten

- **PPO-Implementierung**: Proximal Policy Optimization für stabileres Training
- **Experience Replay**: Speichern und Wiederverwenden vergangener Erfahrungen
- **Multi-Asset Portfolio**: Gleichzeitiges Handeln mehrerer Assets
- **Risikomanagement**: Dynamische Position Sizing basierend auf Volatilität
- **Sentimentanalyse**: Integration von Marktsentiment aus Nachrichten und sozialen Medien
- **Transformer-Modelle**: Implementierung von Attention-basierten Netzwerken für noch bessere Zeitmustererkennung

## Lizenz

MIT

## Mitwirkende

Füge hier deinen Namen hinzu!