import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    Kombiniertes Actor-Critic Netzwerk für Policy-Gradient Reinforcement Learning.

    Verwendet geteilte Feature-Extraktion für Actor und Critic.
    """

    def __init__(self, input_dim, n_actions, hidden_dim=128):
        """
        Initialisiert das Actor-Critic Netzwerk.

        Args:
            input_dim: Dimension des Eingabevektors
            n_actions: Anzahl möglicher Aktionen
            hidden_dim: Dimension der versteckten Schichten
        """
        super(ActorCriticNetwork, self).__init__()

        # Feature-Extraktion (geteilte Schichten)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor-Netzwerk (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        # Critic-Netzwerk (Value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Gewichtsinitialisierung
        self._init_weights()

    def _init_weights(self):
        """Initialisiert die Gewichte des Netzwerks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, state):
        """
        Forward pass durch das Netzwerk.

        Args:
            state: Zustandsvektor

        Returns:
            tuple: (Aktionswahrscheinlichkeiten, Zustandswert)
        """
        # Feature-Extraktion
        shared_features = self.shared_layers(state)

        # Actor: Liefert Wahrscheinlichkeiten für jede Aktion
        logits = self.actor(shared_features)
        action_probs = F.softmax(logits, dim=-1)

        # Critic: Schätzt den Wert des Zustands
        state_value = self.critic(shared_features)

        return action_probs, state_value


class LSTMActorCritic(nn.Module):
    """
    Actor-Critic Netzwerk mit LSTM-Layer für verbesserte temporale Verarbeitung.

    Besonders nützlich für zeitreihenbasierte Entscheidungen im Trading.
    """

    def __init__(self, input_dim, n_actions, hidden_dim=128, lstm_layers=1):
        """
        Initialisiert das LSTM-basierte Actor-Critic Netzwerk.

        Args:
            input_dim: Dimension des Eingabevektors
            n_actions: Anzahl möglicher Aktionen
            hidden_dim: Dimension der versteckten Schichten
            lstm_layers: Anzahl der LSTM-Schichten
        """
        super(LSTMActorCritic, self).__init__()

        # LSTM-Layer für zeitliche Abhängigkeiten
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Feature-Extraktion nach LSTM
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor-Netzwerk (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        # Critic-Netzwerk (Value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Verborgener Zustand (h) und Zellzustand (c) für LSTM
        self.hidden = None

        # Gewichtsinitialisierung
        self._init_weights()

    def _init_weights(self):
        """Initialisiert die Gewichte des Netzwerks."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                else:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)

    def reset_hidden_state(self, batch_size=1):
        """
        Setzt den verborgenen Zustand der LSTM zurück.

        Args:
            batch_size: Größe des Batches
        """
        device = next(self.parameters()).device

        # (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

        self.hidden = (h0, c0)

    def forward(self, state, reset_hidden=False):
        """
        Forward pass durch das Netzwerk.

        Args:
            state: Zustandsvektor
            reset_hidden: Ob der verborgene Zustand zurückgesetzt werden soll

        Returns:
            tuple: (Aktionswahrscheinlichkeiten, Zustandswert)
        """
        batch_size = state.size(0)

        # Zustand umformen, wenn er nicht bereits die richtige Form hat (batch, sequence, features)
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # (batch, 1, features)

        # Hidden State zurücksetzen, falls gewünscht oder nicht vorhanden
        if reset_hidden or self.hidden is None:
            self.reset_hidden_state(batch_size)

        # LSTM-Verarbeitung
        lstm_out, self.hidden = self.lstm(state, self.hidden)

        # Letzten Output des LSTM verwenden
        lstm_out = lstm_out[:, -1, :]

        # Geteilte Feature-Extraktion
        shared_features = self.shared_features(lstm_out)

        # Actor: Liefert Wahrscheinlichkeiten für jede Aktion
        logits = self.actor(shared_features)
        action_probs = F.softmax(logits, dim=-1)

        # Critic: Schätzt den Wert des Zustands
        state_value = self.critic(shared_features)

        return action_probs, state_value