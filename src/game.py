import functools

from src.board import Board, InvalidMove
from src.utils import deepcopy
from src.utils import numpy as np, plt, ArtistAnimation


class Game:
    """
    Tic-Tac-Toe Game Representation

    The class handles the game logic and runs the games.
    It also allows for visualization of a games result using the render method.
    """

    def __init__(self, size, agent0, agent1):
        """
        Constructor

        params:
            - size: Size of the board (one dimension)
            - agent0: Reference to the first agent
            - agent1: Reference to the second agent
        """
        self.board = Board(size, (agent0.i, agent1.i))
        self.agents = (agent0, agent1)

    def run(self):
        """
        Run the game with supplied agents

        returns:
            - Tuple containing result, boards and agents
            - result: One of Board.WON, Board.LOST, Board.DRAW
            - boards: List of intermediate Board states during the game
            - agents: List of references to the playing agents
        """
        boards = []
        while not self.board.is_finished():
            for agent in self.agents:
                move = agent.next_move(deepcopy(self.board))
                try:
                    self.board.execute(move)
                except InvalidMove:
                    print("Invalid move: %s wants to %s" % (agent, move[1]))
                    return Board.WON if agent.i == self.agents[1].i else Board.LOST, boards
                boards.append(deepcopy(self.board))
                if self.board.is_finished():
                    break
        return self.board.get_game_state(self.agents[0]), boards, self.agents

    def render(self, boards):
        """
        Render multiple board states into a single video

        Params:
            boards: numpy array containing board states

        Returns:
            An animation object of matplotlib
        """
        fig, ax = plt.subplots()

        x = self.board.field.shape[0]
        y = self.board.field.shape[1]

        ax.set_xlim(0, x)
        ax.set_ylim(0, y)

        ax.set_xticklabels([])
        ax.set_xticks(np.arange(1, x - 0.5))
        ax.set_yticklabels([])
        ax.set_yticks(np.arange(1, y - 0.5))
        ax.vlines(range(0, x), 0, y, "grey")
        ax.hlines(range(0, y), 0, x, "grey")

        images = []
        images = functools.reduce(lambda images, b: images + [b.render(ax)], boards, [])

        ArtistAnimation(fig, images, interval=1000)
        plt.show()
