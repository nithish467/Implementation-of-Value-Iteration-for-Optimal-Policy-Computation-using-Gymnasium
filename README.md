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

1. Initialize the state-value function \(V(s)\) to zero for all states.
2. Set the discount factor (\(\gamma\)) and convergence threshold (\(\theta\)).
3. For each state, compute the expected value of every possible action using the Bellman Optimality Equation.
4. Update the value of each state with the maximum action value.
5. Repeat the value update process until the maximum change in state values is less than the convergence threshold.
6. After convergence, extract the optimal policy by selecting the action with the highest expected value for each state.
7. Display the optimal state-value function, optimal policy, and the number of iterations required for convergence.





## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "AWWW",
    "WBWW",
    "WWWB",
    "WWWZ"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
env = env.unwrapped

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------
def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            action_values = []

            for a in range(n_actions):
                value = 0

                for prob, next_state, reward, done in env.P[s][a]:
                    value += prob * (reward + gamma * V[next_state])

                action_values.append(value)

            best_value = max(action_values)
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value

        iteration += 1

        if delta < theta:
            break

    # Extract Optimal Policy
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state]
                )

        policy[s] = np.argmax(action_values)

    return V, policy, iteration


# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------
V, policy, iterations = value_iteration(env)

# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: NITHISHKUMAR S")
print("Register Number: 212223240109")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()


```

---

## Output

```text

Name: NITHISHKUMAR S
Register Number: 212223240109
Value Iteration Completed
Number of Iterations: 1

Optimal State-Value Function:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

Optimal Policy:
[['L' 'L' 'L' 'L']
 ['L' 'L' 'L' 'L']
 ['L' 'L' 'L' 'L']
 ['L' 'L' 'L' 'L']]

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

