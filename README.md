# HMM vs Headache: Predicting Headache Occurrences Using Hidden Markov Models

## Overview

This repository contains our analysis on predicting future occurrences of headaches using Hidden Markov Models (HMMs). The dataset comprises a binary sequence of 296 days, indicating the presence (1) or absence (0) of a headache.

## Authors

- **Valeria Insogna**
- **Guglielmo Padula**

## University

Universit`a degli Studi di Trieste

## Dataset

The dataset represents a binary sequence:
- `1`: Day with headache
- `0`: Day without headache

## Objective

To use HMMs and other statistical models to predict the likelihood of experiencing headaches in the future.

## Analysis:

**Autocorrelation:** The first three lags are statistically significant, leading to the decision to model the data as a time series using short memory HMM models.

## General Model Structure:

- \( Z_t \): Markov process (possibly with memory) parametrized by \( \theta \).
- \( X_t | Z_t = z_t \): Follows a Bernoulli distribution governed by a function \( g \).

**Inference:** Used Black Box Expectation Maximization algorithm (via Pyro library).

## Performance Measures:

- BIC and autocorrelation distance in the training data for internal model selection.
- BIC in test data and the autocorrelation distance for the best average model.

## Models Explored:

### Model 1 - Wiener Process:

- \( Z_t \) is modeled as a Wiener process.
- **Performance:** BIC=1277, Autocorrelation distance=0.12.

### Model 2 - Stagional ARIMA:

- \( Z_t \) is modeled as sARIMA(0, 0, 0) × (0, 1, 0)3.
- **Performance:** BIC=1023, Autocorrelation distance=0.08.

### Model 3 - Bayesian Normal Mixture:

- Combines various distributions including Beta, Normal, and InverseGamma.
- **Performance:** BIC=662, Autocorrelation distance=0.12.

### Model 4 - Discrete Time Markov Chain (DTMC):

- \( Z_t \) follows a discrete transition given by matrix \( A \).
- **Performance:** BIC=288, Autocorrelation distance=0.006.

### Model 5 - Mixture of Categorical Variables:

- Combines several distributions like Dirichlet and Beta.
- **Performance:** BIC=252, Autocorrelation distance=0.04.


## Key Findings

- The DTMC Model (Model 4) was found to be the most optimal in predicting headache occurrences.
- The analysis suggests a cyclical pattern: a three-day high likelihood of headaches followed by a three-day low likelihood.

## Tools and Libraries Used

- **Pyro Library** for model implementation and inference
- Other common data science libraries like Pandas, NumPy, and Matplotlib
