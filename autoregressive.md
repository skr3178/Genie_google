# Autoregressive Dynamics Model

An **autoregressive dynamics model** describes how a system evolves over time by
**predicting the next state using previous states (and actions)**.  
The model reuses its own predictions to generate future steps.

---

## Core Idea

At time step `t`, the model predicts the next state:

\[
p(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots)
\]

Most practical systems assume a **Markov property**:

\[
p(s_{t+1} \mid s_t, a_t)
\]

or deterministically:

\[
s_{t+1} = f_\theta(s_t, a_t)
\]

---

## Why "Autoregressive"?

- **Auto** → uses its own previous predictions  
- **Regressive** → predicts the next element in a sequence  

Once trained, the model performs **rollouts**:



Each arrow uses the *same* learned dynamics function.

---

## Deterministic vs Probabilistic Models

### Deterministic
\[
s_{t+1} = f_\theta(s_t, a_t)
\]

- Simple and fast
- No uncertainty modeling
- Errors compound quickly

### Probabilistic
\[
s_{t+1} \sim p_\theta(s_{t+1} \mid s_t, a_t)
\]

- Outputs a distribution (mean, variance)
- Models uncertainty
- More stable long-term rollouts

---

## Latent Autoregressive Dynamics

Instead of predicting raw states, many models predict **latent states**:

\[
z_{t+1} \sim p(z_{t+1} \mid z_t, a_t)
\]

### Advantages
- Lower dimensional
- Easier to model complex observations (images, contacts)
- Enables imagination-based planning

Used in:
- World Models
- Dreamer
- TD-MPC
- MuZero-style planners

---

## Error Accumulation

Autoregressive rollouts suffer from **compounding errors**:

- Small errors early
- Larger errors later
- Can diverge over long horizons

### Common Mitigations
- Short-horizon rollouts
- Stochastic dynamics
- Model ensembles
- Receding-horizon replanning (MPC)

---

## Applications

- Model-based Reinforcement Learning
- Model Predictive Control (MPC)
- Robotics simulation
- Time-series forecasting
- Imagination-based planning

---

## One-Line Summary

> An autoregressive dynamics model predicts future states by repeatedly applying the same learned transition model, using its own past predictions as inputs.

