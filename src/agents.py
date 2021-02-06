from src.utils import numpy as np, deepcopy


class Agent:
    """
    Generic Agent

    Base Class
    """

    def __init__(self, i):
        """
        Constructor:

        params:
            - i: ID of the agent (needs to be unique)
        """
        self.i = i

    def __repr__(self):
        """
        Wrapper for __str__
        """
        return str(self)

    def __str__(self):
        """
        Convert Agent to string using class name and ID
        """
        return "%s %i" % (self.__class__.__name__, self.i)


class RandomAgent(Agent):
    """
    Random Tic-Tac-Toe Agent

    Plays a random move from the list of still allowed moves
    """

    def __init__(self, i):
        """
        Constructor

        params:
            - i: Agent ID
        """
        super().__init__(i)

    def next_move(self, board):
        """
        Compute next move

        selects a move from the boards free list randomly

        params:
            - board: current board state

        returns:
            - the selected move
        """
        moves = board.get_free_list()
        move = moves[np.random.randint(0, len(moves))]
        return self.i, move


class GoodAgent(Agent):
    """
    Good Tic-Tac-Toe Agent

    Tries to enhance its own lines according to length and blocks opponents lines
    """

    def __init__(self, i):
        """
        Constructor

        params:
            - i: Agent ID
            - size: size of the board
        """
        super().__init__(i)
        # Magical weight vector containing the agents logic
        self.w = np.array([100, 8, 4, 2, 1, 32, 16, 8, 4, 2])

    def evaluate(self, board):
        """
        Evaluate a boards quality, see V-function

        params:
            - board: the current board state

        returns:
            A scalar number representing the board quality
        """
        return np.dot(self.w, board.get_lines(self))

    def next_move(self, board):
        """
        Compute next move

        selects the best move based on the resulting boards quality as provided by the evaluate function

        params:
            - board: current board state

        returns:
            - the selected move
        """
        # bestMoveValue = None
        # bestMove = None
        moves = board.get_free_list()
        boards = [deepcopy(board).execute((self.i, move)) for move in moves]
        values = [self.evaluate(board) for board in boards]
        i = np.argmax(values)
        return self.i, moves[i]


class GreedyAgent(GoodAgent):
    """
    Greedy Tic-Tac-Toe Agent

    Tries to extend its own lines without caring for the opponent
    """

    def __init__(self, i):
        """
        Constructor

        params:
            - i: Agent ID
            - size: Size of the board
        """
        super().__init__(i)
        # Magical weight vector containing the agents logic
        self.w = np.array([10000, 1000, 100, 10, 1, 0, 0, 0, 0, 0])


class LearningAgent(GoodAgent):
    """
    Adaptive Tic-Tac-Toe Agent

    Learn a strategy based on the features provided by the board
    """

    def __init__(self, i, eta):
        """
        Constructor

        params:
            - i: Agent ID
            - size: size of the board
        """
        super().__init__(i)
        self.eta = eta
        self.w = np.zeros(2 * 5)

    def learn(self, boards):
        """
        Learn the weight vector based on the boards observed in a game

        TODO: to be implemented by the student

        params:
            - boards: list of boards observed in a game
        returns:
            - learning error
        """
        return 0

    def __str__(self):
        """
        Convert Agent to string using weights and ID
        """
        weights = ""
        for w in self.w:
            weights += f'{w:.2f}, '
        return f"Learning Agent(eta: {self.eta}) ID: {self.i}: {weights[:-2]}"
