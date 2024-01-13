from environment import TicTacToeEnv
from agent import QLearningAgent


def main():
    agent = train_q_learning_agent(verbose=False)


def train_q_learning_agent(episodes=100, verbose=False):
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha=0.1, gamma=0.1, epsilon=0.1)

    for e in range(episodes):
        env.reset()
        state = str(env.state.flatten())
        done = False

        while not done:
            # agent choose action
            available_actions = list(env.available_moves)
            action = agent.choose_action(state, available_actions)

            # agent make action and receives feedback
            old_state = state
            state, reward, done, info = env.step(action)
            state = str(env.state.flatten())
            #  print(state)

            # agent update its beliefs
            old_q_value = agent.get_q_value(old_state, action)
            max_next_q_value = max([agent.get_q_value(state, next_action) for next_action in available_actions])
            next_q_value = (1 - agent.alpha) * old_q_value + agent.alpha * (reward + agent.gamma * max_next_q_value)
            agent.update_q_value(old_state, action, new_value=next_q_value)

        if verbose:
            print(f"Episode {e}: {info}")

    return agent
    

def play():
    """
    Playing TicTacToe 1 vs 1 - Human vs Human
    """
    env = TicTacToeEnv(show=False)
    done = False

    while not done:
        player = env.p1 if env.is_p1_turn else env.p2
        print()
        print(f"Player {player} to play. Selection action [0-8]")
        print(f"Available moves: {env.available_moves}")
        env.render()
        action = int(input())
        state, reward, done, info = env.step(action)

        if done:
            if info["winner"] == 'tie':
                print(f"Tie")
            else:
                print(f"Player {info['winner']} wins!") 


if __name__ == "__main__":
    #  play()
    main()
