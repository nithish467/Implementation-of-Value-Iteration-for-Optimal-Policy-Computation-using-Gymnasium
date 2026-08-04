# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

The objective is to determine the optimal policy for an agent navigating the FrozenLake-v1 environment. The environment consists of safe frozen tiles, holes, a starting position, and a goal state. The agent must maximize the expected cumulative reward by repeatedly updating the value of each state using the Bellman Optimality Equation until convergence and then extracting the optimal policy.

## Software Requirements
- Python 3.x
- Gymnasium
- NumPy
- Visual Studio Code (VS Code) / Jupyter Notebook / Google Colab
- 
## Environment Description

- Environment: FrozenLake-v1 (Gymnasium)
- Grid Size: 4 × 4
- State Space: 16 discrete states
- Action Space: 4 discrete actions
  - Left (0)
  - Down (1)
  - Right (2)
  - Up (3)
- Start State: S
- Goal State: G
- Safe Tiles: F (Frozen)
- Holes: H
- Reward: +1 for reaching the goal, 0 otherwise
- Transition Type: Stochastic (`is_slippery=True`)
- Objective: Find the optimal policy that maximizes the expected cumulative reward while safely reaching the goal.


## MDP Representation

- States (S): 16 states representing the cells of the 4 × 4 FrozenLake grid.
- Actions (A):Four possible actions:
  - Left (0)
  - Down (1)
  - Right (2)
  - Up (3)
- Transition Probability (P): Defines the probability of moving from one state to another after taking an action. In this implementation, `is_slippery=True`, so the transitions are stochastic.
- Reward Function (R):
  - +1 for reaching the Goal (G)
  - 0 for all other transitions
- Discount Factor (γ): 0.99
- Policy (π): A mapping from each state to the action that maximizes the expected cumulative reward.
- Objective: Compute the optimal state-value function and derive the optimal policy using the Value Iteration algorithm.

## Theory

Value Iteration is a Dynamic Programming algorithm used to solve finite Markov Decision Processes (MDPs). It computes the optimal state-value function by repeatedly applying the Bellman Optimality Equation, which updates the value of each state based on the maximum expected reward obtainable from all possible actions. The algorithm continues these updates until the change in state values becomes smaller than a predefined threshold, indicating convergence. Once the optimal value function is obtained, the optimal policy is extracted by selecting the action with the highest expected value for each state. In the FrozenLake-v1 environment, Value Iteration enables the agent to determine the best path from the start state to the goal while maximizing the expected cumulative reward and avoiding holes.



## Algorithm





## Python Program

```python

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------
# Write your code here





```

---

## Output

```text

Number of Iterations: 

Optimal State-Value Function:



Optimal Policy:

```

---

## Result
```text
Write your result here

```
---

## Inference
```text
Write the inference here


```
---

