from src.utils import numpy as np, plt


class InvalidMove(Exception):
    """
    Exception class indicating invalid moves

    Raised if an agent tries to move in a way not according to the rules e.g.:

    returns None
    returns an already occupied cell
    returns a cell outside of the playing field
    """

    def __init__(self, agent, board, move):
        """
        Constructor

        params:
            - culprit: the agent causing the exception
            - state: the current board state without the executed move
            - move: the move causing the exception
        """
        self.culprit = agent
        self.state = board
        self.move = move

    def __repr__(self):
        """
        Wrapper for __str__
        """
        return str(self)

    def __str__(self):
        """
        Conversion to string representation
        """
        return "Illegal Move: %s wants to do %s" % (self.culprit, self.move)


class Board:
    """
    Representation of the game state
    """
    WON = 1
    LOST = 2
    DRAW = 3
    READY = 0
    ONGOING = 4

    def __init__(self, size, agent_ids):
        """
        Constructor

        params:
            - size: Size of the field (single dimension)
            - agentIDs: IDs of the agents playing the game
        """
        self.agent_ids = agent_ids
        self.field = np.zeros((size, size), dtype=np.intp)
        self.lastMove = None
        self.size = size

        # Construction of line templates used for counting lines
        x_template = np.array([(i, 0) for i in range(5)], dtype=np.intp)
        y_template = np.array([(0, i) for i in range(5)], dtype=np.intp)
        x_lines = [x_template + (0, i) for i in range(5)]
        y_lines = [y_template + (i, 0) for i in range(5)]
        cross_lines = [np.array([np.zeros(2, dtype=np.intp) + i for i in range(5)]),
                       np.array([np.array([0, 4], dtype=np.intp) + (i, -i) for i in range(5)])]
        self.lines = []
        self.lines.extend(cross_lines)
        self.lines.extend(x_lines)
        self.lines.extend(y_lines)

    def execute(self, move):
        """
        Execute a move on the board

        params:
            - move: 2 tuple containing agentID and move
        returns:
            - Reference to the modified board
        """
        if self.field[move[1][0], move[1][1]] == 0:
            self.field[move[1][0], move[1][1]] = move[0]
        else:
            raise InvalidMove(move[0], self, move[1])
        self.lastMove = move[1]
        return self

    def __do_it(self, agent_id):
        """
        Helper function to count lines

        params:
            - agentId: ID of the agent lines should be counted for
        returns:
            - list of number of lines in descending order of line length
        """
        line_length = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for line in self.lines:
            length = 0
            for cell in line:
                if self.field[cell[0], cell[1]] == agent_id:
                    length += 1
                    continue
                if not self.field[cell[0], cell[1]] == 0:
                    length = 0
                    break
            line_length[length] += 1
        length_list = np.zeros(self.size)
        for i in range(self.size):
            length_list[i] = line_length[self.size - i]
        return length_list

    def get_lines(self, agent):
        """
        Extracts the current feature vector from the board

        params:
            - agent: the agent defining the perspective of the features
        returns:
            - numpy array containing the number of lines of the agents in descending order of length
              concatenated with the number of lines of the oponnent agent multiplied by -1
        """
        agent_list = self.__do_it(agent.i)
        other_id = [i for i in self.agent_ids if i != agent.i][0]
        other_list = self.__do_it(other_id)
        return np.concatenate([agent_list, -1 * other_list])

    def get_free_list(self):
        """
        Returns the list of still possible moves

        returns:
            - List of moves possible on the current board
        """
        return np.array(np.where(self.field == 0), dtype=np.intp).T

    def is_finished(self):
        """
        Checks if game has ended

        returns:
            - True if game ended, False otherwise
        """
        if len(self.get_free_list()) == 0:
            return True
        if self.__do_it(self.agent_ids[0])[0]:
            return True
        if self.__do_it(self.agent_ids[1])[0]:
            return True
        return False

    def get_game_state(self, agent):
        """
        Returns the current game state

        returns:
            - Board.WON, Board.LOST, Board.DRAW if game ended,
            Board.READY if game not yet started, Board.ONGOING otherwise
        """
        if self.is_finished():
            if self.__do_it(agent.i)[0]:
                return Board.WON
            if not len(self.get_free_list()):
                return Board.DRAW
            return Board.LOST
        if len(self.get_free_list()) == self.size ** 2:
            return Board.READY
        return Board.ONGOING

    def render(self, ax):
        """
        Renders the board to an image

        To be called by Game.render! Not for direct use!

        params:
            - ax: a matplotlib axis object
        returns:
            - a list of drawn patches
        """
        output = []

        for line in np.transpose(np.where(self.field == self.agent_ids[0])):
            output.append(Board.token(ax, line, "red"))

        for line in np.transpose(np.where(self.field == self.agent_ids[1])):
            output.append(Board.token(ax, line, "blue"))

        return output

    @staticmethod
    def token(ax, pos, color):
        """
        Renders a token to the board image

        params:
            - ax: a matplotlib axis object
            - pos: the position to render
            - color: the color of the token
        returns:
            - the patch created by the token
        """
        patch = plt.Circle(pos + (0.5, 0.5), radius=0.4, color=color)
        ax.add_patch(patch)
        return patch
