import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import random

from models.actor_critic import ActorCriticNetwork, LSTMActorCritic


class TradingAgent:
    """
    Trading Agent mit A2C (Advantage Actor-Critic) Algorithmus.

    Der Agent lernt eine Policy und eine Wertfunktion, um optimale
    Handelsentscheidungen zu treffen.
    """

    def __init__(self, input_dim, n_actions, lr=0.001, gamma=0.99,
                 use_lstm=False, hidden_dim=128):
        """
        Initialisiert den Trading Agent.

        Args:
            input_dim: Dimension des Eingabevektors
            n_actions: Anzahl möglicher Aktionen
            lr: Lernrate
            gamma: Discount-Faktor für zukünftige Belohnungen
            use_lstm: Ob ein LSTM-basiertes Netzwerk verwendet werden soll
            hidden_dim: Dimension der versteckten Schichten
        """
        self.gamma = gamma
        self.use_lstm = use_lstm

        # GPU verwenden, falls verfügbar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Netzwerk erstellen (Actor-Critic oder LSTM)
        if use_lstm:
            self.network = LSTMActorCritic(input_dim, n_actions, hidden_dim).to(self.device)
        else:
            self.network = ActorCriticNetwork(input_dim, n_actions, hidden_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Speicher für Episodendaten
        self.reset_memory()

        # Tracking von Training-Statistiken
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }

    def reset_memory(self):
        """Setzt den Speicher für eine neue Episode zurück."""
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.entropies = []

        # LSTM-Zustand zurücksetzen, falls verwendet
        if self.use_lstm:
            self.network.reset_hidden_state()

    def select_action(self, state, training=True):
        """
        Wählt eine Aktion basierend auf dem aktuellen Zustand.

        Args:
            state: Zustandsvektor
            training: Ob der Agent im Trainingsmodus ist

        Returns:
            int: Gewählte Aktion
        """
        # Zustand in Tensor umwandeln
        state_tensor = torch.FloatTensor(state).to(self.device)

        # Zustand umformen, falls nötig
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Forward Pass durch das Netzwerk
        with torch.set_grad_enabled(training):
            action_probs, state_value = self.network(state_tensor)

        # Im Trainingsmodus: Aktion stochastisch wählen
        if training:
            # Epsilon-Greedy Strategie für mehr Exploration
            if random.random() < 0.1:  # 10% Wahrscheinlichkeit für zufällige Aktion
                action = random.randint(0, 2)

                # Erstelle eine künstliche Verteilung für das Logging
                fake_probs = torch.zeros_like(action_probs)
                fake_probs[0, action] = 1.0
                m = Categorical(fake_probs)

                # Speichere Werte für Training
                self.log_probs.append(m.log_prob(torch.tensor([action], device=self.device)))
                self.state_values.append(state_value)
                self.entropies.append(m.entropy())

                return action
            else:
                # Normale stochastische Aktionsauswahl
                m = Categorical(action_probs)
                action = m.sample()

                # Speichere Werte für Training
                self.log_probs.append(m.log_prob(action))
                self.state_values.append(state_value)
                self.entropies.append(m.entropy())

                return action.item()

        # Im Testmodus: Aktion mit höchster Wahrscheinlichkeit wählen
        else:
            return torch.argmax(action_probs).item()

    def update(self, next_state=None, done=False):
        """
        Führt einen Policy-Update durch.

        Args:
            next_state: Nächster Zustand (None falls Episode beendet)
            done: Ob die Episode beendet ist

        Returns:
            float: Gesamtverlust
        """
        # Prüfen, ob genug Daten vorhanden sind
        if len(self.rewards) == 0:
            return 0.0

        # Bootstrap-Value berechnen, wenn Episode nicht beendet ist
        next_value = 0
        if not done and next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value = self.network(next_state_tensor)
                next_value = next_value.item()

        # Returns und Advantages berechnen
        returns = []
        R = next_value

        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Konvertiere in Float32 Tensor für bessere Kompatibilität
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        # Tensoren für Log-Probs, State-Values und Entropies
        log_probs = torch.stack(self.log_probs)
        state_values = torch.stack(self.state_values).squeeze().to(dtype=torch.float32)
        entropies = torch.stack(self.entropies)

        # Advantages berechnen (Returns - State Value)
        advantages = returns - state_values.detach()

        # Verlustfunktionen berechnen

        # Actor (Policy) Loss: maximiere erwarteten Reward
        actor_loss = -(log_probs * advantages).mean()

        # Critic (Value) Loss: minimiere Fehler in der Wertschätzung
        critic_loss = nn.MSELoss()(state_values, returns)

        # Entropy-Regularisierung: fördert Exploration
        entropy_loss = -0.01 * entropies.mean()

        # Gesamtverlust
        loss = actor_loss + 0.5 * critic_loss + entropy_loss

        # Optimiere das Netzwerk
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (verhindert zu große Updates)
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)

        self.optimizer.step()

        # Speichere Statistiken
        self.training_stats['actor_losses'].append(actor_loss.item())
        self.training_stats['critic_losses'].append(critic_loss.item())
        self.training_stats['entropy_losses'].append(entropy_loss.item())
        self.training_stats['total_losses'].append(loss.item())

        # Speicher zurücksetzen
        self.reset_memory()

        return loss.item()

    def store_reward(self, reward):
        """
        Speichert eine Belohnung für das Training.

        Args:
            reward: Belohnung für den letzten Schritt
        """
        # Stelle sicher, dass der Reward ein float32 ist
        self.rewards.append(float(reward))

    def save(self, path):
        """
        Speichert das Modell.

        Args:
            path: Pfad zum Speichern des Modells
        """
        # Stelle sicher, dass der Ordner existiert
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Speichere Modell und Optimierungszustand
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'use_lstm': self.use_lstm
        }, path)

    def load(self, path):
        """
        Lädt ein gespeichertes Modell.

        Args:
            path: Pfad zum gespeicherten Modell
        """
        if not os.path.exists(path):
            print(f"Warnung: Modellpfad existiert nicht: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Prüfe, ob das geladene Modell mit der aktuellen Konfiguration kompatibel ist
            if checkpoint.get('use_lstm', False) != self.use_lstm:
                print(f"Warnung: Gespeichertes Modell hat LSTM={checkpoint.get('use_lstm', False)}, "
                      f"aber der aktuelle Agent hat LSTM={self.use_lstm}")
                return False

            # Lade Modell- und Optimizer-Zustand
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Lade Trainingsstatistiken, falls vorhanden
            if 'training_stats' in checkpoint:
                self.training_stats = checkpoint['training_stats']

            print(f"Modell erfolgreich geladen: {path}")
            return True

        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            return False

    def get_training_stats(self):
        """
        Gibt die Trainingsstatistiken zurück.

        Returns:
            dict: Trainingsstatistiken
        """
        return self.training_stats