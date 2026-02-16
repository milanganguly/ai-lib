![Java](https://img.shields.io/badge/Java-17-blue)
![Build](https://img.shields.io/badge/Build-Gradle-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# ai-lib

A lightweight neural network and automatic differentiation engine written entirely from scratch in Java.

This project implements tensors, reverse-mode autodifferentiation, convolutional neural networks, optimizers, and training pipelines without relying on external deep learning frameworks such as PyTorch or TensorFlow.

The goal is to understand deep learning systems from first principles.

---

## Features

- Custom Tensor implementation
- Reverse-mode automatic differentiation (backpropagation)
- Linear layers
- 2D Convolution
- Max Pooling
- ReLU activation
- Cross-Entropy loss
- Adam optimizer (bias-corrected)
- Model save / load functionality
- End-to-end MNIST training demo

---

## Example Result

Training a small CNN on full MNIST (60,000 training / 10,000 test samples):

Architecture:

```
Conv(1 → 8, 3×3)
ReLU
MaxPool(2×2)
Conv(8 → 16, 3×3)
ReLU
MaxPool(2×2)
Flatten
Linear(400 → 32)
ReLU
Linear(32 → 10)
```

After 1 epoch:

- Test accuracy: ~97.3%

---

## Project Structure

```
src/
 ├── main/java/
 │     ├── milanganguly/engine
 │     ├── milanganguly/nn
 │     ├── milanganguly/optim
 │     └── milanganguly/loss
 │
 ├── main/resources/mnist/
 │     ├── train-images-idx3-ubyte
 │     ├── train-labels-idx1-ubyte
 │     ├── t10k-images-idx3-ubyte
 │     └── t10k-labels-idx1-ubyte
 │
 └── test/java/
       └── MNISTDemo.java
```

---

## Requirements

- Java 17 (recommended)
- Gradle (wrapper included)

Note: Gradle may not support bleeding-edge JVM versions (e.g., Java 25).

---

## Build

From project root:

```bash
./gradlew clean
./gradlew build
```

---

## Run MNIST Demo

After building:

```bash
java -cp build/classes/java/test:build/classes/java/main:build/resources/main MNISTDemo
```

Alternatively, configure the Gradle application plugin and run:

```bash
./gradlew run
```

---

## Model Saving and Loading

Models can be saved and restored:

```java
model.save("model.bin");
model.load("model.bin");
```

The loader verifies parameter counts and tensor shapes to ensure integrity.

---

## Motivation

This project was built to:

- Implement neural networks from first principles
- Understand backpropagation mechanics
- Explore CNN training dynamics
- Build a minimal deep learning engine without external frameworks

---

## License

MIT License
