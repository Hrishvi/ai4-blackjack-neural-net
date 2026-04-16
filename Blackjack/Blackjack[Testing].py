import pickle
import math
import sys
import os


def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))


# =============================================
# HELPER — interpret the output as advice
# =============================================

def interpret(score):
    if score >= 0.6:
        return "HIT   (confidence: high)"
    elif score >= 0.45:
        return "HIT   (confidence: low — borderline call)"
    elif score >= 0.35:
        return "STAND  (confidence: low — borderline call)"
    else:
        return "STAND  (confidence: high)"


# =============================================
# CARD VALUE HELPERS
# =============================================

def normalize_hand(total):
    """Map hand total (8–21) to 0.38–1.00"""
    return round((total - 8) / (21 - 8) * (1.00 - 0.38) + 0.38, 4)

def normalize_dealer(card):
    """Map dealer card (2–10) to 0.20–1.00"""
    return round((card - 2) / (10 - 2) * (1.00 - 0.20) + 0.20, 4)


# =============================================
# LOAD THE MODEL
# Looks inside the "Trained Models" folder.
# =============================================

save_dir = "Trained_Models"

if not os.path.exists(save_dir):
    print(f"No '{save_dir}' folder found. Train a model first.")
    sys.exit()

available = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

if not available:
    print(f"No saved models found in '{save_dir}'. Train a model first.")
    sys.exit()

print("Available models:")
for name in available:
    print(f"  - {name.replace('.pkl', '')}")

model_name = input("\nEnter the model name to load: ").strip()

if model_name == "":
    print("No name entered. Quitting.")
    sys.exit()

load_path = os.path.join(save_dir, model_name + ".pkl")

if not os.path.exists(load_path):
    print(f"Model '{model_name}' not found in '{save_dir}'. Quitting.")
    sys.exit()

with open(load_path, "rb") as f:
    model = pickle.load(f)

print(f"\nModel loaded: {model_name}")

# Pull the weights out of the loaded model
hidden_layer  = model["hidden_layer"]
output_neuron = model["output_neuron"]
bias_hidden   = model["bias_hidden"]
bias_output   = model["bias_output"]

print("\nLoaded Weights and Biases:")
for i, neuron_weights in enumerate(hidden_layer):
    print(f"  hidden_layer[{i}] weights = {neuron_weights[0]:.5f}, {neuron_weights[1]:.5f}  |  bias = {bias_hidden[i]:.5f}")
print(f"  output_neuron weights  = {[f'{w:.5f}' for w in output_neuron]}  |  bias = {bias_output:.5f}")


# =============================================
# TESTING LOOP
# =============================================

while True:
    answer = input("\nDo you want to test the network? [y/n]: ")

    if answer.lower() != "y":
        print("Program ending.")
        break

    your_hand   = int(input("Enter your hand total (8–21): "))
    dealer_card = int(input("Enter dealer's visible card (2–10): "))

    # Clamp to valid range
    your_hand   = max(8,  min(21, your_hand))
    dealer_card = max(2,  min(10, dealer_card))

    # Normalize to 0.0–1.0
    hand_norm   = normalize_hand(your_hand)
    dealer_norm = normalize_dealer(dealer_card)

    inputs = [hand_norm, dealer_norm]

    # --- Forward pass: hidden layer (4 neurons) ---
    hidden_outputs = []
    for i, neuron_weights in enumerate(hidden_layer):
        neuron_sum = sum(inputs[j] * neuron_weights[j] for j in range(2)) + bias_hidden[i]
        hidden_outputs.append(sigmoid(neuron_sum))

    # --- Forward pass: output neuron ---
    output_sum = sum(hidden_outputs[i] * output_neuron[i] for i in range(4)) + bias_output
    output     = sigmoid(output_sum)

    print(f"\n  Your hand : {your_hand}  (normalized: {hand_norm})")
    print(f"  Dealer    : {dealer_card}  (normalized: {dealer_norm})")
    print(f"  Raw score : {output:.4f}")
    print(f"  Decision  : {interpret(output)}")