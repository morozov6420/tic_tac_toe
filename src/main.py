from src.agents import LearningAgent, GreedyAgent, GoodAgent
from src.board import Board
from src.game import Game
from src.utils import numpy as np


def compete(agent1, agent2, n):
    """
    Competition between two agents without learning

    params:
        - agent1: reference to first agent
        - agent2: reference to second agent
        - n: number of games
    returns:
        - Directory with result counts (Boards.DRAW, Boards.WON, Boards.LOST)
    """
    results = {Board.DRAW: 0, Board.WON: 0, Board.LOST: 0}
    for i in range(n):
        result, boards, agents = Game(5, agent1, agent2).run()
        results[result] += 1
    print(f"Results:\n\tWon: {results[Board.WON]}\n\tLost: {results[Board.LOST]}\n\tDraw: {results[Board.DRAW]}")
    return results


def train(agent1, agent2, n):
    """
    Competition between two agents with learning

    params:
        - agent1: reference to first agent
        - agent2: reference to second agent
        - n: number of games
    returns:
        - Directory with result counts (Boards.DRAW, Boards.WON, Boards.LOST)
    """
    agents = [agent1, agent2]
    old_r = None
    d_r = None
    l_agents = [i for i in range(len(agents)) if isinstance(agents[i], LearningAgent)]
    e = []
    results = []
    while True:
        e.append(np.zeros(len(l_agents)))
        results.append({Board.DRAW: 0, Board.WON: 0, Board.LOST: 0})
        for i in range(n):
            result, boards, agents = Game(5, agent1, agent2).run()
            for j in l_agents:
                e[-1][j] = agents[j].learn(boards)
            results[-1][result] += 1
        for i in l_agents:
            print(f"{agents[i]} - Error: {e[-1][i]:.2f}")
        print(f"Results: Won: {results[-1][Board.WON]} Lost: {results[-1][Board.LOST]} Draw: {results[-1][Board.DRAW]}")
        if (e[-1] == 0).any():
            break
        if len(e) >= 2:
            if (np.abs(e[-2] - e[-1]) < 1).any():
                break
            d_r = 0
            for key in results[-2]:
                d_r += abs(results[-2][key] - results[-1][key])
            if d_r <= 1:
                break


if __name__ == '__main__':
    # agent_smith_1 = LearningAgent(1, 5, 0.1)
    # agent_smith_2 = LearningAgent(2, 5, 0.5)
    # compete(agent_smith_1, agent_smith_2, 100)
    # train(agent_smith_1, agent_smith_2, 100)
    # compete(agent_smith_1, agent_smith_2, 100)

    good, greed = GoodAgent(3), GreedyAgent(4)
    game = Game(5, good, greed)
    result, boards, agents = game.run()
    game.render(boards)
    # compete(good, greed, 100)
