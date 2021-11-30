# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        foodCount = successorGameState.getNumFood()
        foodDistance = math.inf
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood.data[i][j]:
                    foodDistance = min(foodDistance, util.manhattanDistance(newPos, (i, j)))
        if foodCount == 0:
            foodDistance = 0
        ghostDistance = math.inf
        for ghost in newGhostStates:
            ghostPos = ghost.configuration.pos
            if ghostPos == newPos:
                return -math.inf
            ghostDistance = min(ghostDistance, util.manhattanDistance(newPos, ghost.configuration.pos))
        if ghostDistance <= foodDistance:
            ghostDistance *= -100
        scaredTime = sum(newScaredTimes)
        score = successorGameState.getScore()
        val = score + scaredTime - foodCount - foodDistance + 1 / ghostDistance + random.random()
        return val

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"



        def getMax(state, searchDepth):
            "Return the maximum value from a state"
            value = -math.inf
            actions = state.getLegalActions(0)
            if len(actions) == 0:
                return self.evaluationFunction(state)
            for action in actions:
                newState = state.generateSuccessor(0, action)
                value = max(value, getMin(newState, 1, searchDepth))
            return value


        def getMin(state, index, searchDepth):
            "Return the minimum value from a state, when the turn is for ghost"
            assert index != 0, "This has to be a ghosts turn."
            value = math.inf
            actions = state.getLegalActions(index)
            if len(actions) == 0:
                return self.evaluationFunction(state)
            for action in actions:
                newState = state.generateSuccessor(index, action)
                if index == newState.getNumAgents() - 1:
                    if searchDepth == 1:
                        value = min(value, self.evaluationFunction(newState))
                    else:
                        value = min(value, getMax(newState, searchDepth - 1))
                else:
                    value = min(value, getMin(newState, index + 1, searchDepth))
            return value



        """
        def getMinMax(state, searchDepth):
            "Return the minmax value from a state, when turn is for agent"
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if state.index == state.getNumAgents():
                "Last ghost's turn"
                searchDepth += 1
                if searchDepth == state.depth:
                    "End of recursion. Return the minimum value"
                    return getMin(state)
                return getMax(state, searchDepth)
                "TODO: Implement case for another round of search"
            state.index = state.index + 1
            "Next ghost turn."
            for state in state.generateSuccessor(state.getLegalActions(state.index)):
                return getMinMax(state, searchDepth)
        """

        answer = None
        value = -math.inf
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            if newState.isWin() or newState.isLose():
                newScore = self.evaluationFunction(newState)
            else:
                newScore = getMin(newState, 1, self.depth)
            if value < newScore:
                value = newScore
                answer = action

        return answer
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def getMax(alpha, beta, state, searchDepth):
            "Return the maximum value from a state"
            value = -math.inf
            actions = state.getLegalActions(0)
            if len(actions) == 0:
                return self.evaluationFunction(state)
            for action in actions:
                newState = state.generateSuccessor(0, action)
                value = max(value, getMin(alpha, beta, newState, 1, searchDepth))
                if alpha == None or value > alpha:
                    alpha = value
                if beta is not None and value > beta:
                    return value
            return value

        def getMin(alpha, beta, state, index, searchDepth):
            "Return the minimum value from a state, when the turn is for ghost"
            assert index != 0, "This has to be a ghosts turn."
            value = math.inf
            actions = state.getLegalActions(index)
            if len(actions) == 0:
                return self.evaluationFunction(state)
            for action in actions:
                newState = state.generateSuccessor(index, action)
                if index == newState.getNumAgents() - 1:
                    if searchDepth == 1:
                        value = min(value, self.evaluationFunction(newState))
                    else:
                        value = min(value, getMax(alpha, beta, newState, searchDepth - 1))
                else:
                    value = min(value, getMin(alpha, beta, newState, index + 1, searchDepth))
                if alpha is not None and value < alpha:
                    return value
                if beta == None or value < beta:
                    beta = value
            return value


        """
        def getMinMax(state, searchDepth):
            "Return the minmax value from a state, when turn is for agent"
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if state.index == state.getNumAgents():
                "Last ghost's turn"
                searchDepth += 1
                if searchDepth == state.depth:
                    "End of recursion. Return the minimum value"
                    return getMin(state)
                return getMax(state, searchDepth)
                "TODO: Implement case for another round of search"
            state.index = state.index + 1
            "Next ghost turn."
            for state in state.generateSuccessor(state.getLegalActions(state.index)):
                return getMinMax(state, searchDepth)
        """

        answer = None
        value = -math.inf
        alpha = None
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            if newState.isWin() or newState.isLose():
                newScore = self.evaluationFunction(newState)
            else:
                newScore = getMin(alpha, None, newState, 1, self.depth)
            if value < newScore:
                value = newScore
                answer = action
            if alpha == None or value > alpha:
                alpha = value

        return answer
        util.raiseNotDefined()
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def getMax(state, searchDepth):
            "Return the maximum value from a state"
            value = -math.inf
            actions = state.getLegalActions(0)
            if len(actions) == 0:
                return self.evaluationFunction(state)
            for action in actions:
                newState = state.generateSuccessor(0, action)
                value = max(value, getMin(newState, 1, searchDepth))
            return value


        def getMin(state, index, searchDepth):
            "Return the minimum value from a state, when the turn is for ghost"
            assert index != 0, "This has to be a ghosts turn."
            actions = state.getLegalActions(index)
            if len(actions) == 0:
                return self.evaluationFunction(state)
            value = 0
            for action in actions:
                newState = state.generateSuccessor(index, action)
                if index == newState.getNumAgents() - 1:
                    if searchDepth == 1:
                        value += self.evaluationFunction(newState) / len(actions)
                    else:
                        value += getMax(newState, searchDepth - 1) / len(actions) + 1
                else:
                    value += getMin(newState, index + 1, searchDepth) / len(actions) + 1
            return value

        answer = None
        value = -math.inf
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)

            if newState.isWin() or newState.isLose():
                newScore = self.evaluationFunction(newState)
            else:
                newScore = getMin(newState, 1, self.depth)
            if value < newScore:
                value = newScore
                answer = action

        return answer
        "*** YOUR CODE HERE ***"


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    score: score
    - 1 / (ghostDistance + random.random(): take inverse of ghostDistance but add random to avoid divide by 0.
        The closer the distance the worse the evaluation becomes.
    - foodDistance: the distance to the closest next food. The further the distance the worse it becomes.
        If there are no food left, becomes 0.
    - len(foodList): also take into account how many food we have left.
    """
    "*** YOUR CODE HERE ***"

    pacPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    ghostPosition = []
    for i in range(1, currentGameState.getNumAgents()):
        ghostPosition.append(currentGameState.getGhostPosition(i))
    ghostDistance = 0
    for pos in ghostPosition:
        ghostDistance += util.manhattanDistance(pacPosition, pos)
    if ghostStates[0].scaredTimer > 0:
        ghostDistance *= -1
        score += 10
    foodList = currentGameState.getFood().asList()
    foodDistance = math.inf
    if len(foodList) == 0:
        foodDistance = 0
    for food in foodList:
        foodDistance = min(foodDistance, util.manhattanDistance(pacPosition, food))
    if currentGameState.isWin():
        score += 100
    return score - 1 / (ghostDistance + random.random()) - foodDistance - len(foodList)

# Abbreviation
better = betterEvaluationFunction
