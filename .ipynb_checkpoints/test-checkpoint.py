import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random

# --------------------------
# Configuración del experimento
# --------------------------
mlflow.set_experiment("modelo_proy_final_5")

# Modelo ficticio de tensores
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Datos simulados
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# --------------------------
# Emulación de entrenamiento
# --------------------------
import random

with mlflow.start_run(run_name="entrenamiento_run") as run:
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 20
    max_progress = 0.4  # Máximo 40%
    base_progress = np.linspace(0.05, max_progress, epochs)  # progreso base

    for epoch in range(epochs):
        # Forward pass ficticio
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass ficticio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métrica simulada de "accuracy" con más variabilidad
        noise = random.uniform(-0.02, 0.05)  # fluctuación mayor
        simulated_accuracy = max(0.0, min(base_progress[epoch] + noise, max_progress))

        # Log en MLflow con fluctuación en loss también
        loss_noise = loss.item() * random.uniform(0.95, 1.05)
        mlflow.log_metric("loss", loss_noise, step=epoch)
        mlflow.log_metric("accuracy", simulated_accuracy, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_noise:.4f} - Accuracy: {simulated_accuracy:.3f}")
        time.sleep(5.5)  # simula tiempo de entrenamiento


    # Guardar modelo final (opcional)
    mlflow.pytorch.log_model(model, "simple_net_model")

print("Simulación finalizada. Revisa MLflow UI para ver métricas y modelo.")
