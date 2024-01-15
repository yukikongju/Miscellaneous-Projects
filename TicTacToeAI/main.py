from environment import TicTacToeEnv
from agent import QLearningAgent, SARSAAgent


def main():
    #  agent = train_q_learning_agent(verbose=False)
    agent = train_sarsa_agent(verbose=False)


def train_sarsa_agent(episodes=100, verbose=False):
    env = TicTacToeEnv()
    agent = SARSAAgent(alpha=0.1, gamma=0.1, epsilon=0.1)

    # train the agent
    for e in range(episodes):
        env.reset()
        available_actions = list(env.available_moves)
        old_state = str(env.state.flatten())
        old_action = agent.choose_action(old_state, available_actions)
        done = False

        while not done:
            # agent choose action
            available_actions = list(env.available_moves)

            # agent make action and receives feedback
            action = agent.choose_action(old_state, available_actions)
            state, reward, done, info = env.step(action)
            state = str(env.state.flatten())
            #  print(state)

            # agent update its beliefs
            agent.update_q_value(old_state=old_state, state=state, 
                                 old_action=old_action,
                                 action=action, reward=reward, 
                                 available_actions=available_actions)

            old_action = action
            old_state = state

        if verbose:
            print(f"Episode {e}: {info}")

    return agent


def train_q_learning_agent(episodes=100, verbose=False):
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha=0.1, gamma=0.1, epsilon=0.1)

    # train the agent
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
            agent.update_q_value(old_state=old_state, state=state, 
                                 action=action, reward=reward, 
                                 available_actions=available_actions)

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
