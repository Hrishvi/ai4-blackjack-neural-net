import random   # Used to generate random starting weights for the network
import math     # Used for the math.exp() function inside sigmoid
import pickle   # Used to save the trained model to a file
import os       # Used to create folders and build file paths


# =============================================
# TRAINING DATA
# =============================================
# Each row is one training example in the format:
#   [your_hand (normalized), dealer_card (normalized), correct_answer]
#
# "Normalized" means the values have been scaled to a 0.0–1.0 range
# so the network can process them consistently. See the Testing script
# for the exact normalization formulas.
#
# The correct answer (target) is:
#   1.00 = HIT  (the right move is to take another card)
#   0.00 = STAND (the right move is to stay with your current hand)
#
# Reading the data below as raw Blackjack logic:
#   Row 1: hand≈8,  dealer≈2  → HIT   (low hand, weak dealer — hit aggressively)
#   Row 2: hand≈9,  dealer≈7  → HIT   (still low, dealer has a decent card)
#   Row 3: hand≈11, dealer≈4  → STAND (risky to hit; dealer likely to bust)
#   Row 4: hand≈12, dealer≈9  → HIT   (medium hand, strong dealer — hit)
#   Row 5: hand≈15, dealer≈5  → STAND (dealer weak; let them bust)
#   Row 6: hand≈16, dealer≈10 → HIT   (dealer very strong; must try)
#   Row 7: hand≈18, dealer≈7  → STAND (solid hand; don't risk busting)
#   Row 8: hand≈20, dealer≈10 → STAND (near-perfect hand; never hit)

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
# NETWORK CONFIGURATION
# =============================================

n_inputs = 2
# The network takes exactly 2 inputs per example:
#   input[0] = your normalized hand total
#   input[1] = the dealer's normalized visible card

hidden_neurons = 4
# The hidden layer has 4 neurons.
# These are the "thinking" neurons in the middle of the network.
# They detect patterns in the inputs before passing results to the output.
# More neurons = more capacity to learn complex patterns,
# but also more risk of overfitting on small datasets.

learning_rate = 0.05
# Controls how big a step the network takes when adjusting weights.
# Too high → learning is unstable and overshoots the answer.
# Too low  → learning is very slow and may get stuck.
# 0.05 is a safe middle-ground for a small network like this.

epochs = 200000
# One "epoch" = one full pass through all 8 training examples.
# After 200,000 epochs, the network has seen each example 200,000 times.
# This is how neural networks learn: repetition and gradual correction.


# =============================================
# HELPER FUNCTIONS
# =============================================

def sigmoid(x):
    # The sigmoid function "squashes" any number into the range (0.0, 1.0).
    # This is used as the activation function for every neuron.
    # It lets the network output probabilities rather than raw unbounded numbers.
    #
    # Formula: sigmoid(x) = 1 / (1 + e^(-x))
    #
    # Examples:
    #   sigmoid(0)   = 0.5   (perfectly uncertain)
    #   sigmoid(5)   ≈ 0.99  (strongly positive)
    #   sigmoid(-5)  ≈ 0.01  (strongly negative)
    #
    # We clamp x to [-500, 500] first to avoid math overflow errors
    # when e^(-x) becomes astronomically large or small.
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))


def init_weight():
    # Returns a random float between -1.0 and +1.0.
    # All weights in the network start as random values.
    # Starting random ensures neurons don't all learn the same thing
    # (if all weights started at 0, all neurons would be identical forever).
    # The range [-1, 1] keeps initial outputs near sigmoid's midpoint (0.5).
    return random.uniform(-1, 1)


# =============================================
# INITIALIZE THE NETWORK WEIGHTS AND BIASES
# =============================================

# hidden_layer is a list of 4 neurons.
# Each neuron has 2 weights — one for each input.
# So hidden_layer[i][j] = the weight connecting input j to hidden neuron i.
#
# Visualized:
#   input[0] ──w[0][0]──┐
#                        ├──→ hidden neuron 0
#   input[1] ──w[0][1]──┘
#
#   input[0] ──w[1][0]──┐
#                        ├──→ hidden neuron 1
#   input[1] ──w[1][1]──┘
#   ... and so on for neurons 2 and 3.

hidden_layer = [[init_weight() for _ in range(n_inputs)] for _ in range(hidden_neurons)]
# Result: a 4×2 grid of random weights, e.g.:
# [[-0.42,  0.71],   ← neuron 0's weights
#  [ 0.13, -0.88],   ← neuron 1's weights
#  [ 0.55,  0.22],   ← neuron 2's weights
#  [-0.09,  0.67]]   ← neuron 3's weights

output_neuron = [init_weight() for _ in range(hidden_neurons)]
# The single output neuron has 4 weights — one for each hidden neuron.
# It combines all hidden neuron outputs into a final single prediction.
# e.g.: [0.34, -0.61, 0.88, -0.15]

bias_hidden = [0.0] * hidden_neurons
# Each hidden neuron has its own bias value, all starting at 0.
# A bias shifts a neuron's activation up or down independently of the inputs.
# Think of it like the y-intercept in y = mx + b — it lets the neuron
# "fire" even when inputs are near zero.

bias_output = 0.0
# The output neuron also has a single bias, starting at 0.


# =============================================
# TRAINING LOOP
# =============================================
# This is the core of the learning process.
# For every epoch, we loop through every training example and:
#   1. Run the inputs through the network (forward pass) to get a prediction
#   2. Compare that prediction to the correct answer (calculate error)
#   3. Adjust all weights slightly to reduce the error (backpropagation)
# After enough repetitions, the weights converge to values that work well.

for epoch in range(epochs):

    total_loss = 0  # Accumulates the total error across all examples this epoch.
                    # We use this only for printing progress — it doesn't affect training.

    for your_hand, dealer_hand, target in training_data:
        # Unpack each training example into its three parts:
        #   your_hand   = normalized hand total (e.g. 0.38)
        #   dealer_hand = normalized dealer card (e.g. 0.20)
        #   target      = correct answer (1.0 = hit, 0.0 = stand)

        inputs = [your_hand, dealer_hand]
        # Bundle inputs into a list for easy indexing during the forward pass.


        # ── FORWARD PASS: Hidden Layer ──────────────────────────────────────
        # Each hidden neuron computes a weighted sum of both inputs, adds its
        # bias, then passes the result through sigmoid to produce an output.
        #
        # Formula for neuron i:
        #   sum_i = (input[0] × weight[i][0]) + (input[1] × weight[i][1]) + bias_i
        #   output_i = sigmoid(sum_i)

        hidden_outputs = []  # Will hold the output of each of the 4 hidden neurons.

        for i in range(hidden_neurons):
            # Compute the weighted sum for hidden neuron i
            s = sum(inputs[j] * hidden_layer[i][j] for j in range(n_inputs)) + bias_hidden[i]
            # Pass through sigmoid to get this neuron's output (a value 0–1)
            hidden_outputs.append(sigmoid(s))


        # ── FORWARD PASS: Output Neuron ─────────────────────────────────────
        # The output neuron takes all 4 hidden outputs, applies its own weights
        # and bias, then passes through sigmoid for the final prediction.
        #
        # Formula:
        #   output_sum = Σ(hidden_outputs[i] × output_neuron[i]) + bias_output
        #   output = sigmoid(output_sum)

        output_sum = sum(hidden_outputs[i] * output_neuron[i] for i in range(hidden_neurons)) + bias_output
        output = sigmoid(output_sum)
        # 'output' is now a number between 0 and 1:
        #   close to 1.0 → network thinks: HIT
        #   close to 0.0 → network thinks: STAND


        # ── CALCULATE ERROR ─────────────────────────────────────────────────
        # How wrong was the network's prediction?
        #
        # error = target - output
        #   Positive error → prediction was too low (network should output more)
        #   Negative error → prediction was too high (network should output less)

        error = target - output

        # Squared error penalizes large mistakes more than small ones.
        # We sum it up for the epoch-level loss display.
        total_loss += error**2


        # ── BACKPROPAGATION: Output Layer ────────────────────────────────────
        # Now we figure out how much to adjust the output neuron's weights.
        #
        # The gradient tells us the direction and magnitude of the adjustment.
        # It combines:
        #   - the error (how wrong we were)
        #   - the derivative of sigmoid: output × (1 - output)
        #     (this scales adjustments based on how "confident" the output was —
        #      neurons near 0.5 get bigger updates; saturated neurons get smaller ones)
        #
        # Formula: output_gradient = error × output × (1 - output)

        output_gradient = error * output * (1 - output)

        # Save the current output weights BEFORE updating them.
        # We'll need the original values when backpropagating into the hidden layer.
        old_output_weights = output_neuron[:]

        # Adjust each output weight by a small step in the right direction.
        # Formula: new_weight = old_weight + learning_rate × gradient × hidden_output_i
        # The hidden_output_i term means: weights connected to more active neurons
        # are updated more aggressively (they "contributed more" to the error).
        for i in range(hidden_neurons):
            output_neuron[i] += learning_rate * output_gradient * hidden_outputs[i]

        # Adjust the output bias by the same gradient (biases have no input to multiply by).
        bias_output += learning_rate * output_gradient


        # ── BACKPROPAGATION: Hidden Layer ────────────────────────────────────
        # Now propagate the error signal backwards into the hidden layer.
        # Each hidden neuron gets a share of the blame proportional to its
        # contribution to the output error.

        for i in range(hidden_neurons):
            # How much did hidden neuron i contribute to the output error?
            # We use the OLD output weight (before it was updated above) times
            # the output gradient to assign each hidden neuron its share of blame.
            hidden_error = output_gradient * old_output_weights[i]

            # Apply the sigmoid derivative for this hidden neuron.
            # Same logic as before: scale the update based on neuron confidence.
            hidden_gradient = hidden_error * hidden_outputs[i] * (1 - hidden_outputs[i])

            # Adjust each weight connecting an input to this hidden neuron.
            # inputs[j] scales the update: inputs that were larger contributed more,
            # so their weights get adjusted more.
            for j in range(n_inputs):
                hidden_layer[i][j] += learning_rate * hidden_gradient * inputs[j]

            # Adjust this hidden neuron's bias.
            bias_hidden[i] += learning_rate * hidden_gradient


    # ── PROGRESS REPORT ─────────────────────────────────────────────────────
    # Every 10,000 epochs, print the total squared error across all examples.
    # As training progresses, this number should steadily decrease toward 0.
    # If it stops decreasing or increases, the learning rate may be too high.
    if epoch % 10000 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss:.6f}")


# =============================================
# SAVE THE TRAINED MODEL
# =============================================
# After training, we save the weights and biases to a .pkl file.
# "Pickling" serializes Python objects to binary so they can be
# reloaded later exactly as they were — no retraining needed.

save_dir = "Trained_Models"
os.makedirs(save_dir, exist_ok=True)
# Create the save folder if it doesn't already exist.
# exist_ok=True means no error is raised if the folder is already there.

model_name = input("\nEnter model name: ").strip()
# Ask the user what to call this model.
# .strip() removes any accidental leading/trailing spaces.

if not model_name:
    model_name = "model"
# If the user just pressed Enter with no input, default to "model".

save_path = os.path.join(save_dir, model_name + ".pkl")
# Build the full file path, e.g.: "Trained_Models/my_model.pkl"

# Bundle all the learned values into a single dictionary.
# This is what gets saved — the exact structure the Testing script expects.
model_data = {
    "hidden_layer":  hidden_layer,   # 4×2 list of hidden neuron weights
    "output_neuron": output_neuron,  # list of 4 output neuron weights
    "bias_hidden":   bias_hidden,    # list of 4 hidden neuron biases
    "bias_output":   bias_output,    # single float: the output neuron's bias
}

with open(save_path, "wb") as f:
    # Open the file in write-binary mode ("wb") and pickle the dictionary into it.
    pickle.dump(model_data, f)

print(f"\nModel saved to: {save_path}")
