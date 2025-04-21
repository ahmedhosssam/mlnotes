# What is Regularization?

- **Regularization** is a set of techniques used to reduce the complexity of a machine learning model and prevent it from over fitting the training data.
- Regularization adds a **penalty** to the model’s **loss function**, discouraging it from fitting the training data too perfectly.
- Regularization works by adding a **regularization term** to the loss function (which the model tries to minimize).

## Example

This is the original loss function (for example):

$$
\text{cost}(W) = \frac{1}{2N} \sum_{i=1}^{N} \left( y(X^n, W) - t^n \right)^2
$$

After adding one of the regularization types (ridge, for example), it becomes:

$$
\text{cost}(W) = \frac{1}{2N} \sum_{i=1}^{N} \left( y(X^i, W) - t^i \right)^2 + \sum_{i=1}^{M} \left( \frac{\lambda}{2} W_i^2 \right)
$$

The new term is the **Regularization Term**.

Given one of the weights \( W_j \), the partial derivative will be:

$$
\frac{\partial \text{cost}(W)}{\partial W_j} = \frac{1}{N} \sum_{i=1}^{N} \left( y(X^i, W) - t^n \right)^2 \cdot X_j^i + \lambda W_j
$$

# Ridge Regression

- We also call it L2 Regularization.
- Adds the squared magnitude of all weights to the loss function:

  $$
  \textbf{Penalty} = \sum_{i=1}^{M} w_j^2
  $$

- It encourages **smaller weights**.
- It keeps all features, but shrinks their influence.
- **Notice** that we started with **i=1**, which means we don't penalize the intercept at \( W_0 \).

# Lasso Regression

- Adds the **absolute value** of weights to the loss function:

  $$
  \textbf{Penalty} = \sum_{j=1}^{M} \left| w_j \right|
  $$

- Encourages **sparsity** — sets some weights to zero.
- Can be used for **feature selection**.
