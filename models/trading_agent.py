import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import random
import time

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

        # GPU-Optimierungen wenn verfügbar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # Optimierungen für CUDA
            torch.backends.cudnn.benchmark = True
            if self.use_lstm:
                # LSTM kann mit deterministic=True effizienter sein auf CUDA
                torch.backends.cudnn.deterministic = True

            print(f"TradingAgent wird auf {self.device} mit CUDA-Optimierungen initialisiert")
        else:
            print(f"TradingAgent wird auf {self.device} initialisiert")

        # Netzwerk erstellen (Actor-Critic oder LSTM)
        if use_lstm:
            self.network = LSTMActorCritic(input_dim, n_actions, hidden_dim).to(self.device)
        else:
            self.network = ActorCriticNetwork(input_dim, n_actions, hidden_dim).to(self.device)

        # Optimizer mit grad clipping
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

        # Modell-Performance-Statistiken
        self.cuda_stats = {
            'forward_time': [],
            'backward_time': [],
            'memory_usage': []
        } if torch.cuda.is_available() else None

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
        # GPU-Monitoring Variablen initialisieren
        start_memory = None
        # GPU-Monitoring, falls aktiviert
        if self.cuda_stats and training and random.random() < 0.05:  # Nur für 5% der Aufrufe tracken
            if torch.cuda.is_available():
                start_memory = torch.cuda.memory_allocated() / 1024 ** 2

        # Zustand in Tensor umwandeln
        state_tensor = torch.FloatTensor(state).to(self.device)

        # Zustand umformen, falls nötig
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Zeit für Forward-Pass messen, falls CUDA-Tracking aktiviert
        start_time = time.time() if self.cuda_stats and training else None

        # Forward Pass durch das Netzwerk
        with torch.set_grad_enabled(training):
            action_probs, state_value = self.network(state_tensor)

        # CUDA-Statistiken aktualisieren
        if self.cuda_stats and training and start_time and random.random() < 0.05:
            forward_time = time.time() - start_time
            self.cuda_stats['forward_time'].append(forward_time * 1000)  # in ms

            if torch.cuda.is_available() and start_memory is not None:  # Prüfen ob start_memory initialisiert wurde
                current_memory = torch.cuda.memory_allocated() / 1024 ** 2
                self.cuda_stats['memory_usage'].append(current_memory - start_memory)

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

        # CUDA-Monitoring
        start_time = time.time() if self.cuda_stats else None

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

        if state_values.shape != returns.shape:
            if len(state_values.shape) == 0:  # Skalar
                state_values = state_values.unsqueeze(0)  # Füge eine Dimension hinzu
            elif len(returns.shape) == 0:  # Skalar
                returns = returns.unsqueeze(0)  # Füge eine Dimension hinzu
        # Critic (Value) Loss: minimiere Fehler in der Wertschätzung
        critic_loss = nn.MSELoss()(state_values, returns)

        # Entropy-Regularisierung: fördert Exploration
        entropy_loss = -0.01 * entropies.mean()

        # Gesamtverlust
        loss = actor_loss + 0.5 * critic_loss + entropy_loss

        # Optimiere das Netzwerk
        self.optimizer.zero_grad()

        # Zeit für Backward-Pass messen
        backward_start = time.time() if self.cuda_stats else None

        loss.backward()

        # Gradient Clipping (verhindert zu große Updates)
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.1)

        self.optimizer.step()

        # CUDA-Statistiken für Backward-Pass aktualisieren
        if self.cuda_stats and backward_start:
            backward_time = time.time() - backward_start
            self.cuda_stats['backward_time'].append(backward_time * 1000)  # in ms

        # Speichere Statistiken
        self.training_stats['actor_losses'].append(actor_loss.item())
        self.training_stats['critic_losses'].append(critic_loss.item())
        self.training_stats['entropy_losses'].append(entropy_loss.item())
        self.training_stats['total_losses'].append(loss.item())

        # Speicher zurücksetzen
        self.reset_memory()

        # CUDA-Synchronisierung, um genaue Timings zu gewährleisten
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Gesamtzeit für Update berechnen
        if self.cuda_stats and start_time:
            total_time = time.time() - start_time
            if len(self.cuda_stats['forward_time']) > 100:  # Begrenze Statistik-Größe
                # Entferne älteste Einträge
                self.cuda_stats['forward_time'] = self.cuda_stats['forward_time'][-100:]
                self.cuda_stats['backward_time'] = self.cuda_stats['backward_time'][-100:]
                self.cuda_stats['memory_usage'] = self.cuda_stats['memory_usage'][-100:]

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
        save_dict = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'use_lstm': self.use_lstm
        }

        # Füge CUDA-Statistiken hinzu, falls vorhanden
        if self.cuda_stats:
            cuda_avg_stats = {
                'avg_forward_ms': np.mean(self.cuda_stats['forward_time']) if self.cuda_stats['forward_time'] else 0,
                'avg_backward_ms': np.mean(self.cuda_stats['backward_time']) if self.cuda_stats['backward_time'] else 0,
                'avg_memory_mb': np.mean(self.cuda_stats['memory_usage']) if self.cuda_stats['memory_usage'] else 0
            }
            save_dict['cuda_stats'] = cuda_avg_stats

        # Speichern mit Fehlerbehandlung
        try:
            torch.save(save_dict, path)
            print(f"Modell erfolgreich gespeichert: {path}")

            if self.cuda_stats and self.cuda_stats['forward_time']:
                print(f"CUDA Performance: Forward {np.mean(self.cuda_stats['forward_time']):.2f}ms, "
                      f"Backward {np.mean(self.cuda_stats['backward_time']):.2f}ms, "
                      f"Memory Usage: {np.mean(self.cuda_stats['memory_usage']):.2f}MB")

        except Exception as e:
            print(f"Fehler beim Speichern des Modells: {e}")

    def load(self, path):
        """
        Lädt ein gespeichertes Modell.

        Args:
            path: Pfad zum gespeicherten Modell

        Returns:
            bool: True wenn erfolgreich, False sonst
        """
        if not os.path.exists(path):
            print(f"Warnung: Modellpfad existiert nicht: {path}")
            return False

        try:
            # Lade auf CPU, falls keine GPU verfügbar ist
            map_location = self.device
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)

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

            # Zeige CUDA-Statistiken, falls vorhanden
            if 'cuda_stats' in checkpoint:
                cuda_stats = checkpoint['cuda_stats']
                print(f"Geladene CUDA-Performance: Forward {cuda_stats.get('avg_forward_ms', 0):.2f}ms, "
                      f"Backward {cuda_stats.get('avg_backward_ms', 0):.2f}ms, "
                      f"Memory Usage: {cuda_stats.get('avg_memory_mb', 0):.2f}MB")

            print(f"Modell erfolgreich geladen: {path}")
            return True

        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_training_stats(self):
        """
        Gibt die Trainingsstatistiken zurück.

        Returns:
            dict: Trainingsstatistiken
        """
        stats = self.training_stats.copy()

        # Füge CUDA-Statistiken hinzu, falls vorhanden
        if self.cuda_stats:
            stats['cuda'] = {
                'forward_time_ms': np.mean(self.cuda_stats['forward_time']) if self.cuda_stats['forward_time'] else 0,
                'backward_time_ms': np.mean(self.cuda_stats['backward_time']) if self.cuda_stats[
                    'backward_time'] else 0,
                'memory_usage_mb': np.mean(self.cuda_stats['memory_usage']) if self.cuda_stats['memory_usage'] else 0
            }

        return stats

    def print_cuda_info(self):
        """Gibt Informationen zur CUDA-Nutzung aus."""
        if not torch.cuda.is_available():
            print("CUDA ist nicht verfügbar.")
            return

        print("\n=== CUDA-Informationen ===")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Speichernutzung: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB / "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        if self.cuda_stats:
            if self.cuda_stats['forward_time']:
                print(f"Durchschnittliche Forward-Zeit: {np.mean(self.cuda_stats['forward_time']):.2f} ms")
            if self.cuda_stats['backward_time']:
                print(f"Durchschnittliche Backward-Zeit: {np.mean(self.cuda_stats['backward_time']):.2f} ms")
            if self.cuda_stats['memory_usage']:
                print(
                    f"Durchschnittliche Speichernutzung pro Operation: {np.mean(self.cuda_stats['memory_usage']):.2f} MB")

        # Zeige Modellstatistiken an
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"Modellparameter: {total_params:,} gesamt, {trainable_params:,} trainierbar")
        print("=" * 30)