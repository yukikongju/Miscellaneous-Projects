from environment import TicTacToeEnv


def main():
    env = TicTacToeEnv(show=False)
    done = False

    while not done:
        player = env.p1 if env.is_p1_turn else env.p2
        print()
        print(f"Player {player} to play. Selection action [0-8]")
        env.render()
        action = int(input())
        state, reward, done, info = env.step(action)

        if done:
            if info["winner"] == 'tie':
                print(f"Tie")
            else:
                print(f"Player {info['winner']} wins!") 


if __name__ == "__main__":
    main()
