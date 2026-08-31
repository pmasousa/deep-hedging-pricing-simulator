# Public Deep Hedging & Pricing Simulator

## Overview
This project is an open-source, simplified showcase of advanced quantitative finance research. Designed as a legally shareable counterpart to proprietary thesis work, it leverages differential machine learning to price European or Barrier options and calculate Greeks efficiently.

## Objectives
- Demonstrate specialized quantitative finance and AI knowledge in a public repository.
- Implement differential machine learning for accurate pricing and risk management.
- Benchmark deep learning approaches against classical quantitative models.

## Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow or PyTorch
- **Quantitative Finance:** QuantLib, custom differential ML implementations
- **Simulation:** Standard Monte Carlo simulators (e.g., using NumPy/CuPy)
- **Visualization:** Matplotlib, Plotly

## Technical Architecture
1. **Path Simulation Module:**
   - Monte Carlo engine simulating underlying asset paths (e.g., Geometric Brownian Motion or Heston model).
2. **Classical Pricing Engine:**
   - Baseline pricing and Greek calculation using Black-Scholes and standard QuantLib routines.
3. **Differential ML Model:**
   - A neural network trained not only on option payoffs (prices) but also on the pathwise derivatives (Greeks).
   - Custom loss function combining value error and derivative error.
4. **Benchmarking Suite:**
   - Tools to compare execution time, memory usage, and accuracy of the Differential ML model vs. the Classical Pricing Engine.

## Project Roadmap
1. Build the Monte Carlo path simulator.
2. Implement classical Black-Scholes pricing and Greeks for baselining.
3. Construct the PyTorch/TensorFlow network and the custom differential loss function.
4. Generate training data and train the Differential ML model.
5. Create benchmarking scripts and a comprehensive Jupyter Notebook for presentation.
