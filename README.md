![Java](https://img.shields.io/badge/Java-17-blue)
![Build](https://img.shields.io/badge/Build-Gradle-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# ai-lib

ai-lib is a deep learning framework built entirely from first principles in pure Java.

It implements tensors, reverse-mode automatic differentiation, convolutional neural networks, optimizers, and training pipelines without relying on external ML frameworks such as PyTorch or TensorFlow.

The purpose of this project is to deeply understand how modern neural networks work internally — from tensor operations to gradient propagation and parameter updates.

---

## Example Result

Training a small CNN on full MNIST (60,000 training / 10,000 test samples) achieved ~97.3% test accuracy after 1 epoch.

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

## Features

- Custom Tensor implementation  
- Reverse-mode automatic differentiation (backpropagation)  
- Linear layers  
- 2D Convolution  
- Max Pooling  
- ReLU activation  
- Cross-Entropy and MSE loss  
- Adam optimizer (bias-corrected implementation)  
- Model save / load functionality  
- End-to-end MNIST training demo  

---

## Why Java?

- Demonstrate low-level understanding without high-level ML frameworks  
- Explore performance trade-offs in a managed runtime  
- Show that neural networks are not framework-dependent  
- Build strong fundamentals in autodifferentiation and tensor math  

---

## Engine Design

ai-lib is structured into modular components that mirror the core building blocks of modern deep learning frameworks.

### Core Components

**Tensor**
- Multidimensional array implementation
- Optional gradient tracking (`requiresGrad`)
- Stores data and accumulated gradients
- Supports basic tensor operations used in forward computation

**Autograd Engine**
- Reverse-mode automatic differentiation
- Builds a dynamic computation graph during forward pass
- Propagates gradients in reverse topological order during backward pass
- Enables gradient accumulation for parameter updates

**Neural Network Modules**
- Linear
- Conv2D
- MaxPool
- ReLU
- Sigmoid
- Flatten

Modules expose:
- `forward(Tensor input)`
- Parameter access for optimizers

**Loss Functions**
- Cross-Entropy
- Mean Squared Error (MSE)

**Optimizers**
- Adam (bias-corrected implementation)
- Parameter update based on accumulated gradients

### Training Flow

1. Forward pass builds computation graph  
2. Loss is computed  
3. `Autograd.backward(loss)` performs reverse-mode differentiation  
4. Optimizer updates parameters  
5. Gradients are cleared  

This design keeps the engine minimal while preserving conceptual similarity to larger frameworks.

---

## Limitations

ai-lib is designed primarily as an educational project.

Current limitations include:

- CPU-only (no GPU acceleration)
- No multi-threading or parallelization
- Limited operator coverage compared to full ML frameworks
- No automatic batching abstraction
- No JIT compilation or graph optimization
- Not optimized for production-scale training

The focus of this project is clarity and understanding rather than raw performance.

---

## Future Improvements

Potential areas for future development:

- GPU acceleration support (e.g., JNI bindings or CUDA backend)
- Multi-threaded CPU execution
- Additional layers (BatchNorm, Dropout, Residual blocks)
- More activation functions (LeakyReLU, GELU, Tanh)
- Expanded optimizer support (SGD with momentum, RMSProp)
- Automatic batching utilities
- Gradient checking utilities for debugging
- Performance benchmarking suite
- Dataset abstraction layer beyond MNIST
- Improved model serialization format

These improvements would extend the framework while preserving its educational focus.

---

## Example Usage

Small network example:

```java
Sequential seq = new Sequential(
    new Linear(10, 32),
    new ReLU(),
    new Linear(32, 5),
    new Sigmoid()
);

Adam optimizer = new Adam(seq, 0.001f);
MSE lossFn = new MSE();

Tensor data = new Tensor(data, true, new int[]{10});
Tensor target = new Tensor(target, false, new int[]{5});

Tensor output = seq.forward(data);
Tensor loss = lossFn.forward(output, target);

optimizer.zeroGrad();
Autograd.backward(loss);
optimizer.step();
```

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
