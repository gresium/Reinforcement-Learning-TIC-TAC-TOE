# Reinforcement Learning Tic-Tac-Toe
A Python implementation of Reinforcement Learning for the classic game Tic-Tac-Toe.
An intelligent agent learns optimal strategies through Q-learning by playing repeated games, improving performance through trial and error.

# Objectives
Apply reinforcement learning to a simple turn-based environment
Teach the agent to play Tic-Tac-Toe via self-play
Visualize learning progress through rewards and win rates
Demonstrate Q-learning in a discrete state-action space

# Methods
Algorithm: Q-learning
Learning Principle: Temporal Difference (TD) updates
Update Rule:
Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s’, :)) − Q(s, a)]
Exploration Policy: ε-greedy (balances exploration and exploitation)
Key Parameters
α (learning rate): controls update speed
γ (discount factor): weighs future rewards
ε (exploration rate): randomness in decision-making
episodes: number of training games
Repository Contents
reinforcement_tic_tac_toe.py — main training script
Reinforcement Learning TIC-TAC_TOE.ipynb — interactive notebook version
README.md — documentation

# QuickStart
Create environment
python -m venv .venv
Activate environment
macOS/Linux: source .venv/bin/activate
Windows: .venv\Scripts\activate
Install dependencies
pip install numpy matplotlib
Run training
python reinforcement_tic_tac_toe.py
Open notebook for visualization
jupyter lab or jupyter notebook
Workflow
Initialize Q-table for all possible board states
Simulate episodes of Tic-Tac-Toe (agent vs. opponent/self)
Update Q-values based on game results
Gradually reduce ε to shift from exploration to exploitation
Evaluate agent performance and visualize results
Possible Extensions
Add human vs. AI mode
Visualize Q-values for board positions
Experiment with different reward functions
Extend to other games (e.g., Connect Four)

# Author
Developed by Gresa Hisa (@gresium) — AI & Cybersecurity Engineer
