# 🌋 Landslide Susceptibility Mapping in Mount Elgon Region

## Project Overview

This repository hosts the code and resources for developing a **Landslide Susceptibility Map (LSM)** for the highly vulnerable Mount Elgon region, utilizing an advanced **Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM)** deep learning architecture.

The project combines static topographic and soil data with dynamic, time-series rainfall data to create a robust spatial-temporal model capable of predicting the probability of landslide occurrence across the study area.

---

## 🔬 Methodology: CNN-LSTM Approach

The CNN-LSTM hybrid model is employed to leverage its strengths:

1.  **CNN (Convolutional Neural Network):** Used for efficiently extracting complex **spatial features** and their correlations from the static environmental layers (Slope, TWI, Aspect, Soil).
2.  **LSTM (Long Short-Term Memory):** Used for processing and understanding the **temporal patterns** in the rainfall time-series data, crucial for accurately modeling rainfall-induced failures.

### Key Predictor Variables ($\text{X}$)

| Category | Feature | Source/Derivation | Type |
| :--- | :--- | :--- | :--- |
| **Static Spatial** | Digital Elevation Model (DEM) | SRTM 30m | Raster |
| **Static Spatial** | Slope, Aspect, TWI | Derived from DEM | Raster |
| **Static Spatial** | Soil Classification | HWSD (Harmonized World Soil Database) | Raster |
| **Dynamic Temporal** | Antecedent Rainfall Index (ARI) | CHIRPS/GEE Time Series | Time Series |

---

## 📂 Repository Structure

The project is organized into a clean, modular structure for easy navigation and execution:
