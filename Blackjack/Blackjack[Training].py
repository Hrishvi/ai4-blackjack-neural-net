import random
import math
import pickle
import os

# =============================================
# TRAINING DATA (keep yours or improved version)
# =============================================

training_data = [
    [0.38, 0.20, 1.00],
    [0.45, 0.60, 1.00],
    [0.55, 0.30, 0.00],
    [0.60, 0.80, 1.00],
    [0.70, 0.40, 0.00],
    [0.75, 0.85, 1.00],
    [0.85, 0.50, 0.00],
    [0.95, 0.90, 0.00],
]

# =============================================
# CONFIG
# =============================================

n_inputs = 2
hidden_neurons = 4

learning_rate = 0.05
epochs = 200000

# =============================================
# HELPERS
# =============================================

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))

def init_weight():
    return random.uniform(-1, 1)

# =============================================
# INIT NETWORK
# =============================================

hidden_layer = [[init_weight() for _ in range(n_inputs)] for _ in range(hidden_neurons)]
output_neuron = [init_weight() for _ in range(hidden_neurons)]

bias_hidden = [0.0] * hidden_neurons
bias_output = 0.0

# =============================================
# TRAINING LOOP
# =============================================

for epoch in range(epochs):

    total_loss = 0

    for your_hand, dealer_hand, target in training_data:

        inputs = [your_hand, dealer_hand]

        # ── Forward pass ──
        hidden_outputs = []
        for i in range(hidden_neurons):
            s = sum(inputs[j] * hidden_layer[i][j] for j in range(n_inputs)) + bias_hidden[i]
            hidden_outputs.append(sigmoid(s))

        output_sum = sum(hidden_outputs[i] * output_neuron[i] for i in range(hidden_neurons)) + bias_output
        output = sigmoid(output_sum)

        # ── Error ──
        error = target - output
        total_loss += error**2

        # ── Backprop (output) ──
        output_gradient = error * output * (1 - output)

        old_output_weights = output_neuron[:]

        for i in range(hidden_neurons):
            output_neuron[i] += learning_rate * output_gradient * hidden_outputs[i]

        bias_output += learning_rate * output_gradient

        # ── Backprop (hidden) ──
        for i in range(hidden_neurons):
            hidden_error = output_gradient * old_output_weights[i]
            hidden_gradient = hidden_error * hidden_outputs[i] * (1 - hidden_outputs[i])

            for j in range(n_inputs):
                hidden_layer[i][j] += learning_rate * hidden_gradient * inputs[j]

            bias_hidden[i] += learning_rate * hidden_gradient

    # ── Print progress ──
    if epoch % 10000 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss:.6f}")

# =============================================
# SAVE MODEL (MATCH TESTING CODE FORMAT)
# =============================================

save_dir = "Trained_Models"
os.makedirs(save_dir, exist_ok=True)

model_name = input("\nEnter model name: ").strip()
if not model_name:
    model_name = "model"

save_path = os.path.join(save_dir, model_name + ".pkl")

model_data = {
    "hidden_layer": hidden_layer,
    "output_neuron": output_neuron,
    "bias_hidden": bias_hidden,
    "bias_output": bias_output,
}

with open(save_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to: {save_path}")