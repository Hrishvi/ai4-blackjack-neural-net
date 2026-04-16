import pickle  # Used to load the saved model file (.pkl)
import math    # Used for math.exp() inside the sigmoid function
import sys     # Used to call sys.exit() and quit the program gracefully
import os      # Used to check if folders/files exist and build file paths


# =============================================
# SIGMOID ACTIVATION FUNCTION
# =============================================

def sigmoid(x):
    # Squashes any number into the range (0.0, 1.0).
    # This is the same function used during training.
    # We need it here to replicate the exact same forward pass the network learned.
    #
    # Why clamp to [-500, 500]?
    # math.exp(-500) is effectively 0, and math.exp(500) is astronomically large.
    # Python would throw an OverflowError without this safety clamp.
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))


# =============================================
# INTERPRET THE OUTPUT AS ADVICE
# =============================================

def interpret(score):
    # Translates the network's raw output (a float between 0 and 1)
    # into a human-readable Hit/Stand recommendation with a confidence level.
    #
    # The thresholds below divide the 0–1 range into 4 zones:
    #
    #   0.60 – 1.00 → Strongly predicts HIT   → High confidence HIT
    #   0.45 – 0.60 → Leans toward HIT         → Low confidence HIT
    #   0.35 – 0.45 → Leans toward STAND       → Low confidence STAND
    #   0.00 – 0.35 → Strongly predicts STAND  → High confidence STAND
    #
    # The "borderline" band (0.35–0.60) covers genuinely ambiguous situations
    # where either play could be reasonable depending on full game context.

    if score >= 0.6:
        return "HIT   (confidence: high)"
    elif score >= 0.45:
        return "HIT   (confidence: low — borderline call)"
    elif score >= 0.35:
        return "STAND  (confidence: low — borderline call)"
    else:
        return "STAND  (confidence: high)"


# =============================================
# INPUT NORMALIZATION HELPERS
# =============================================
# Neural networks work best when all inputs live in the same numeric range.
# Raw card values (e.g. hand total = 17, dealer card = 9) are on very
# different scales, so we map them to consistent ranges using linear scaling.
#
# General formula:
#   normalized = (value - min) / (max - min) × (new_max - new_min) + new_min

def normalize_hand(total):
    # Maps a hand total (8 to 21) into the range 0.38 to 1.00.
    #
    # Why start at 8? In Blackjack, hands below 8 are never worth discussing
    # (you'd always hit with anything under 8). So 8 is our practical minimum.
    #
    # Why not map to 0.00–1.00? The lower bound of 0.38 was chosen to match
    # the training data's lowest hand value, keeping predictions in-distribution.
    #
    # Example: normalize_hand(8)  → 0.38  (lowest possible hand)
    #          normalize_hand(21) → 1.00  (Blackjack / perfect hand)
    #          normalize_hand(15) → ~0.69 (mid-range hand)
    return round((total - 8) / (21 - 8) * (1.00 - 0.38) + 0.38, 4)


def normalize_dealer(card):
    # Maps the dealer's visible card (2 to 10) into the range 0.20 to 1.00.
    #
    # Why start at 2? In Blackjack, the lowest possible visible card is a 2.
    # (Aces are handled separately in full Blackjack — we simplify here.)
    #
    # Why 0.20 as the lower bound? Again, this matches the training data's
    # minimum dealer value, keeping the network in familiar territory.
    #
    # Example: normalize_dealer(2)  → 0.20  (weakest dealer card)
    #          normalize_dealer(10) → 1.00  (strongest dealer card)
    #          normalize_dealer(7)  → ~0.80 (fairly strong dealer card)
    return round((card - 2) / (10 - 2) * (1.00 - 0.20) + 0.20, 4)


# =============================================
# LOAD THE SAVED MODEL
# =============================================
# Before we can run the network, we need to reload the weights and biases
# that were saved during training. They live in the "Trained_Models" folder
# as .pkl (pickle) files.

save_dir = "Trained_Models"
# The folder name where all trained models are stored.

if not os.path.exists(save_dir):
    # If the folder doesn't exist at all, no model has ever been trained.
    # Tell the user and quit — there's nothing to load.
    print(f"No '{save_dir}' folder found. Train a model first.")
    sys.exit()

available = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]
# os.listdir() returns every file/folder name inside save_dir.
# We filter to only files ending in ".pkl" — those are our saved models.
# Result: a list like ["Blackjack.pkl", "my_model.pkl"]

if not available:
    # The folder exists but contains no .pkl files — training hasn't been run yet.
    print(f"No saved models found in '{save_dir}'. Train a model first.")
    sys.exit()

# Show the user which models are available to load.
print("Available models:")
for name in available:
    # Strip the ".pkl" extension so the display is cleaner (just the model name).
    print(f"  - {name.replace('.pkl', '')}")

model_name = input("\nEnter the model name to load: ").strip()
# Ask the user to type the name of the model they want to use.
# .strip() removes any accidental whitespace around the input.

if model_name == "":
    # If the user pressed Enter without typing anything, quit cleanly.
    print("No name entered. Quitting.")
    sys.exit()

load_path = os.path.join(save_dir, model_name + ".pkl")
# Reconstruct the full file path from the folder name and the model name.
# e.g.: "Trained_Models/Blackjack.pkl"

if not os.path.exists(load_path):
    # The file path doesn't exist — the user probably mistyped the name.
    print(f"Model '{model_name}' not found in '{save_dir}'. Quitting.")
    sys.exit()

with open(load_path, "rb") as f:
    # Open the file in read-binary mode ("rb") — pickle files are binary.
    model = pickle.load(f)
    # pickle.load() deserializes the file back into the Python dictionary
    # that was originally saved during training.

print(f"\nModel loaded: {model_name}")


# ── Unpack the model dictionary into separate variables ──────────────────────
# The dictionary has exactly 4 keys, matching what the training script saved.

hidden_layer  = model["hidden_layer"]
# A 4×2 list: the weights connecting inputs to hidden neurons.
# hidden_layer[i][j] = weight from input j to hidden neuron i.

output_neuron = model["output_neuron"]
# A list of 4 weights: one per hidden neuron, connecting to the output.

bias_hidden   = model["bias_hidden"]
# A list of 4 bias values, one per hidden neuron.

bias_output   = model["bias_output"]
# A single float: the output neuron's bias.


# ── Print the loaded weights for transparency ─────────────────────────────────
# This lets the user see exactly what the network learned.
# Weights close to 0 have little influence; large positive/negative values
# mean the network found that connection very important.

print("\nLoaded Weights and Biases:")
for i, neuron_weights in enumerate(hidden_layer):
    print(f"  hidden_layer[{i}] weights = {neuron_weights[0]:.5f}, {neuron_weights[1]:.5f}  |  bias = {bias_hidden[i]:.5f}")
print(f"  output_neuron weights  = {[f'{w:.5f}' for w in output_neuron]}  |  bias = {bias_output:.5f}")


# =============================================
# INTERACTIVE TESTING LOOP
# =============================================
# Repeatedly ask the user for a hand scenario, run the forward pass,
# and display the network's recommendation. Loop until they say no.

while True:

    answer = input("\nDo you want to test the network? [y/n]: ")

    if answer.lower() != "y":
        # .lower() handles both "N" and "n" as a quit signal.
        print("Program ending.")
        break  # Exit the while loop and end the program.


    # ── Get user input ───────────────────────────────────────────────────────
    your_hand   = int(input("Enter your hand total (8–21): "))
    dealer_card = int(input("Enter dealer's visible card (2–10): "))

    # Clamp values to valid Blackjack ranges.
    # If the user types 25 for their hand, max() brings it down to 21.
    # If they type 1 for the dealer card, max() brings it up to 2.
    # This prevents the normalization formulas from producing out-of-range results.
    your_hand   = max(8,  min(21, your_hand))
    dealer_card = max(2,  min(10, dealer_card))


    # ── Normalize inputs ──────────────────────────────────────────────────────
    # Convert raw card values into the 0–1 range the network was trained on.
    # Without this step, the network would receive completely unfamiliar numbers
    # and produce meaningless outputs.
    hand_norm   = normalize_hand(your_hand)
    dealer_norm = normalize_dealer(dealer_card)

    inputs = [hand_norm, dealer_norm]
    # Bundle normalized inputs into a list for the forward pass.


    # ── FORWARD PASS: Hidden Layer ────────────────────────────────────────────
    # Reproduce the same computation the network performed during training.
    # For each hidden neuron:
    #   1. Multiply each input by the neuron's corresponding weight
    #   2. Sum those products together
    #   3. Add the neuron's bias
    #   4. Pass the result through sigmoid to get the neuron's output

    hidden_outputs = []  # Stores the output (0–1) of each hidden neuron.

    for i, neuron_weights in enumerate(hidden_layer):
        # neuron_weights is the list of 2 weights for hidden neuron i.
        # We compute the weighted sum across all inputs, then add the bias.
        neuron_sum = sum(inputs[j] * neuron_weights[j] for j in range(2)) + bias_hidden[i]
        # Apply sigmoid to squash the sum into a 0–1 range.
        hidden_outputs.append(sigmoid(neuron_sum))


    # ── FORWARD PASS: Output Neuron ───────────────────────────────────────────
    # Combine all 4 hidden outputs into a single final prediction.
    # Each hidden output is multiplied by its corresponding output weight,
    # summed together, then the bias is added before the final sigmoid.

    output_sum = sum(hidden_outputs[i] * output_neuron[i] for i in range(4)) + bias_output
    output     = sigmoid(output_sum)
    # 'output' is now between 0 and 1:
    #   → values near 1.0 mean the network is confident the right move is HIT
    #   → values near 0.0 mean the network is confident the right move is STAND
    #   → values near 0.5 mean the network is genuinely unsure


    # ── Display the result ────────────────────────────────────────────────────
    print(f"\n  Your hand : {your_hand}  (normalized: {hand_norm})")
    # Shows the raw hand total and its normalized equivalent.

    print(f"  Dealer    : {dealer_card}  (normalized: {dealer_norm})")
    # Shows the dealer's card and its normalized equivalent.

    print(f"  Raw score : {output:.4f}")
    # The network's raw output, displayed to 4 decimal places.
    # This is the uninterpreted probability-like value before we apply thresholds.

    print(f"  Decision  : {interpret(output)}")
    # The human-readable recommendation with confidence level,
    # derived from the raw score using the interpret() function above.
