# blackjack-neural-net

> A beginner-friendly neural network built from scratch in Python — trained to advise Hit or Stand in Blackjack.

---

## Introduction

Welcome! This project is a tiny, hand-crafted **neural network** that learns to play Blackjack — and it's built without any fancy AI libraries. No TensorFlow. No PyTorch. Just plain Python and a whole lot of math explained step by step.

If you've ever wondered *"how does an AI actually learn?"* — this is the perfect place to find out. The network takes two inputs (your hand total and the dealer's visible card), thinks it over through a hidden layer of neurons, and outputs a recommendation: **Hit** or **Stand**.

It's simple by design. And that's exactly what makes it so powerful for learning.

---

## Features

- **Pure Python** — no external ML libraries required, just the standard library
- **Train your own model** — run the training script and watch the network learn in real time
- **Interactive testing** — load any saved model and query it with your own Blackjack scenarios
- **Confidence levels** — the network doesn't just say Hit/Stand, it tells you *how sure* it is
- **Saved models** — trained weights are saved as `.pkl` files so you can reload and reuse them
- **Readable code** — every step is commented and explained, making it ideal for learning

---

## Getting Started

You only need **Python 3** installed. No pip installs, no setup headaches.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/blackjack-neural-net.git
cd blackjack-neural-net
```

### 2. Train the model

```bash
python Blackjack_Training_.py
```

The network will train over 200,000 epochs, printing its loss every 10,000 steps. When it finishes, it will ask you to give your model a name — type anything you like (e.g. `my_first_model`) and it saves automatically into a `Trained_Models/` folder.

### 3. Test the model

```bash
python Blackjack_Testing_.py
```

You'll be prompted to enter your hand total (8–21) and the dealer's visible card (2–10). The network will respond with a decision like:

```
  Your hand : 14  (normalized: 0.6431)
  Dealer    : 7   (normalized: 0.8250)
  Raw score : 0.6183
  Decision  : HIT   (confidence: high)
```

That's it — you're running a neural network! 🎉

---

## Who It's For

This project is made for:

- **Complete beginners** curious about how AI works under the hood
- **Students** learning about neural networks, backpropagation, or machine learning concepts
- **Hobbyist programmers** who want a fun, bite-sized Python project
- **Tinkerers** who love modifying training data and seeing what changes

No background in AI or data science is required. If you can run a Python script, you're ready.

---

## 💡 Why This Repo?

Most neural network tutorials either wave their hands at the math, or dump you into a library like TensorFlow with no explanation of what's actually happening inside.

This project does neither. Every weight, every bias, every forward pass and backpropagation step is written out in plain Python — so you can *read* what the network is doing, not just *run* it.

**What you'll understand after exploring this project:**

- What a neuron actually does (it's just multiplication and addition!)
- How a network "learns" by adjusting weights through backpropagation
- Why we use a sigmoid activation function
- How input normalization helps training stability
- What loss means and why we want it to go down

This is the project that makes the theory click.

---

## Project Structure

```
blackjack-neural-net/
│
├── Blackjack_Training_.py    # Train the neural network from scratch
├── Blackjack_Testing_.py     # Load a trained model and test it interactively
├── Trained_Models/           # Your saved .pkl model files live here
│   └── Blackjack.pkl         # A pre-trained example model included to get started
└── README.md
```

---

## Contributing

Contributions are warmly welcome! Here are some ideas if you'd like to expand the project:

- Add more training examples to improve accuracy
- Experiment with more hidden neurons or layers
- Visualize the training loss over time with matplotlib
- Add support for Aces and soft hands
- Build a simple terminal UI for a full Blackjack game

To contribute, fork the repo, make your changes on a new branch, and open a Pull Request. All skill levels welcome — even fixing a typo counts!

---

## License

This project is released under the [MIT License](LICENSE) — free to use, modify, and share.

---

*Built with curiosity, pure Python, and a healthy respect for the dealer's face-down card.*
