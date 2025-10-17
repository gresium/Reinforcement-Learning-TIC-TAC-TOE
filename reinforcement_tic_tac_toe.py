# tic_tac_toe_rl.py
import numpy as np
import pickle
from collections import defaultdict
from typing import Optional, Tuple, List

BOARD_ROWS = 3
BOARD_COLS = 3

class State:
    """
    Environment: Tic-Tac-Toe board.
    Board values: 1 for Player 1 (X), -1 for Player 2 (O), 0 empty.
    """
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.p1 = p1
        self.p2 = p2
        self.is_end = False
        self.board_hash: Optional[str] = None
        self.player_symbol = 1  # Player 1 starts

    # unique string for current board state
    def get_hash(self) -> str:
        self.board_hash = str(self.board.reshape(BOARD_ROWS * BOARD_COLS))
        return self.board_hash

    # check terminal and return outcome: 1 win, -1 loss, 0 tie, None not end
    def winner(self) -> Optional[int]:
        # rows
        for i in range(BOARD_ROWS):
            row_sum = np.sum(self.board[i, :])
            if row_sum == 3:
                self.is_end = True
                return 1
            if row_sum == -3:
                self.is_end = True
                return -1
        # cols
        for j in range(BOARD_COLS):
            col_sum = np.sum(self.board[:, j])
            if col_sum == 3:
                self.is_end = True
                return 1
            if col_sum == -3:
                self.is_end = True
                return -1
        # diagonals
        diag1 = sum(self.board[i, i] for i in range(BOARD_COLS))
        diag2 = sum(self.board[i, BOARD_COLS - 1 - i] for i in range(BOARD_COLS))
        if diag1 == 3 or diag2 == 3:
            self.is_end = True
            return 1
        if diag1 == -3 or diag2 == -3:
            self.is_end = True
            return -1

        # tie
        if len(self.available_positions()) == 0:
            self.is_end = True
            return 0

        # not end
        self.is_end = False
        return None

    def available_positions(self) -> List[Tuple[int, int]]:
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def update_state(self, position: Tuple[int, int]):
        self.board[position] = self.player_symbol
        # switch player
        self.player_symbol = -1 if self.player_symbol == 1 else 1

    def give_reward(self):
        result = self.winner()
        if result == 1:          # p1 win
            self.p1.feed_reward(1.0)
            self.p2.feed_reward(0.0)
        elif result == -1:       # p2 win
            self.p1.feed_reward(0.0)
            self.p2.feed_reward(1.0)
        else:                    # tie
            self.p1.feed_reward(0.5)
            self.p2.feed_reward(0.5)

    def reset(self):
        self.board[:] = 0
        self.is_end = False
        self.board_hash = None
        self.player_symbol = 1

    def show(self):
        s = {1: "X", -1: "O", 0: " "}
        print("\n---------")
        for i in range(BOARD_ROWS):
            row = "|".join(s[self.board[i, j]] for j in range(BOARD_COLS))
            print(row)
            if i < BOARD_ROWS - 1:
                print("-----")
        print("---------")


class Player:
    def __init__(self, name: str, exp_rate: float = 0.3, lr: float = 0.2, gamma: float = 0.9):
        self.name = name
        self.states: List[str] = []          # record of state hashes this episode
        self.lr = lr                          # learning rate
        self.exp_rate = exp_rate              # epsilon
        self.gamma = gamma                    # discount factor
        self.decay = 0.9995                   # small epsilon decay per move
        self.symbol = 1                       # set later by environment
        self.states_value = defaultdict(float)  # Q(s)

    def choose_action(self, positions: List[Tuple[int, int]], current_board: np.ndarray):
        # ε-greedy
        if np.random.rand() <= self.exp_rate:
            idx = np.random.choice(len(positions))
            return positions[idx]

        # pick action with max Q(s')
        value_max = -1e9
        action = positions[0]
        for p in positions:
            next_board = current_board.copy()
            next_board[p] = self.symbol
            next_hash = str(next_board.reshape(BOARD_ROWS * BOARD_COLS))
            value = self.states_value[next_hash]
            if value >= value_max:
                value_max = value
                action = p
        return action

    def add_state(self, state_hash: str):
        self.states.append(state_hash)

    def feed_reward(self, reward: float):
        # backpropagate through the episode
        for st in reversed(self.states):
            self.states_value[st] += self.lr * (self.gamma * reward - self.states_value[st])
            reward = self.states_value[st]    # bootstrap
        self.states = []
        # gentle exploration decay over time
        self.exp_rate *= self.decay

    def reset(self):
        self.states = []

    def save_policy(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(dict(self.states_value), f)

    def load_policy(self, file: str):
        with open(file, "rb") as f:
            data = pickle.load(f)
        self.states_value.update(data)


class HumanPlayer:
    def __init__(self, name: str):
        self.name = name
        self.symbol = -1  # By default, human plays O

    def choose_action(self, positions, current_board):
        print("Available moves:", positions)
        while True:
            row = int(input("Enter row (0/1/2): "))
            col = int(input("Enter col (0/1/2): "))
            if (row, col) in positions:
                return (row, col)
            print("Invalid move, try again.")

    def add_state(self, s): pass
    def feed_reward(self, r): pass
    def reset(self): pass


def train(episodes=50000, save_as: Optional[str] = "policy_p1.pkl"):
    p1 = Player("p1")
    p2 = Player("p2")
    st = State(p1, p2)
    print(f"Training for {episodes} self-play games...")
    for ep in range(episodes):
        if (ep + 1) % 5000 == 0:
            print("Episode:", ep + 1)
        while True:
            positions = st.available_positions()
            p1_action = p1.choose_action(positions, st.board)
            st.update_state(p1_action)
            st_hash = st.get_hash()
            p1.add_state(st_hash)
            win = st.winner()
            if win is not None:
                st.give_reward()
                p1.reset(); p2.reset()
                st.reset()
                break

            positions = st.available_positions()
            p2_action = p2.choose_action(positions, st.board)
            st.update_state(p2_action)
            st_hash = st.get_hash()
            p2.add_state(st_hash)
            win = st.winner()
            if win is not None:
                st.give_reward()
                p1.reset(); p2.reset()
                st.reset()
                break

    if save_as:
        p1.save_policy(save_as)
        print(f"Saved Player 1 policy to {save_as}")
    return p1, p2


def play_human(policy_file: Optional[str] = "policy_p1.pkl"):
    # load trained AI as Player 1 (X). Human plays O.
    ai = Player("AI", exp_rate=0.0)
    if policy_file:
        ai.load_policy(policy_file)
    human = HumanPlayer("You")
    st = State(ai, human)
    while True:
        # AI move
        positions = st.available_positions()
        action = ai.choose_action(positions, st.board)
        st.update_state(action)
        st.show()
        win = st.winner()
        if win is not None:
            if win == 1:
                print("AI wins!")
            elif win == -1:
                print("You win!")
            else:
                print("It's a tie.")
            break

        # Human move
        positions = st.available_positions()
        action = human.choose_action(positions, st.board)
        st.update_state(action)
        st.show()
        win = st.winner()
        if win is not None:
            if win == 1:
                print("AI wins!")
            elif win == -1:
                print("You win!")
            else:
                print("It's a tie.")
            break


if __name__ == "__main__":
    # 1) Train once (or skip if you already have a saved policy)
    # train(episodes=50000, save_as="policy_p1.pkl")

    # 2) Then play vs the trained AI:
    # play_human("policy_p1.pkl")
    pass

