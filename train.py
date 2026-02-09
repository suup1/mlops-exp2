import time
import random

print("=== Experiment 3: Jenkins ML Training Started ===")

# Simulated training loop
for epoch in range(1, 6):
    loss = random.uniform(0.2, 0.6)
    print(f"Epoch {epoch} - loss: {loss:.4f}")
    time.sleep(1)

print("=== Training Completed Successfully ===")
