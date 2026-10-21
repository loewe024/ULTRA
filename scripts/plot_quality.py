import torch
import matplotlib.pyplot as plt

# Tensor laden
data = torch.load("./output/retarget_export_080_quality/sub10_largebox_050_xyz1p000000-1p000000-1p000000_080_080_080.pt")  # Shape: (T, 3)

# In NumPy konvertieren
data = data.cpu().numpy()

# Zeitlicher Verlauf der drei Variablen plotten
plt.figure(figsize=(12, 5))
plt.plot(data[:, 0], label="Reward")
plt.plot(data[:, 1], label="Foot Error")
plt.plot(data[:, 2], label="Object Error")

plt.xlabel("Frame")
plt.ylabel("Wert")
plt.title("Zeitlicher Verlauf der drei Variablen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()