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
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        def distanceToNearestFood():
            """
            Calculates the Manhattan distance to the nearest food.
            """
            foodList = newFood.asList()
            if not foodList:  # No food left
                return 0
            return min(manhattanDistance(newPos, food) for food in foodList)

        def distanceFromGhosts():
            """
            Calculates a penalty based on the Manhattan distance to active ghosts.
            """
            penalty = 0
            for i, ghostState in enumerate(newGhostStates):
                ghostPos = ghostState.getPosition()
                dist = manhattanDistance(newPos, ghostPos)
                if newScaredTimes[i] == 0:  # Only consider active ghosts
                    penalty += max(0, 4 - dist) * 100  # High penalty for proximity
            return penalty

        if successorGameState.isWin():
            return 99999  # High reward for winning
        if successorGameState.isLose():
            return -99999  # High penalty for losing

        # Combine distance to food and ghost penalties for evaluation
        foodDistance = distanceToNearestFood()
        ghostPenalty = distanceFromGhosts()

        # Lower score for stopping
        stopPenalty = 10 if action == Directions.STOP else 0

        return (
            successorGameState.getScore()
            - ghostPenalty
            - stopPenalty
            + (1 / (foodDistance + 1)) * 10  # Higher score for being closer to food
        )

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        # Call the unified minimax function and get the best action
        return self.minimax(gameState, 0, 0)[1]

    def minimax(self, gameState, depth, agentID):
        """
        Unified minimax function handling both max and min agents.
        Returns a tuple of (score, action).
        """
        # Check terminal state or max depth
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentID)
        if not legalActions:  # No actions available
            return self.evaluationFunction(gameState), None

        # Initialize for max/min agent
        if agentID == 0:  # Pacman's turn (maximizer)
            value, best_action = float('-inf'), Directions.STOP
        else:  # Ghosts' turn (minimizer)
            value, best_action = float('inf'), None

        # Iterate over legal actions
        for action in legalActions:
            successor = gameState.generateSuccessor(agentID, action)

            # Determine the next agent and depth
            nextAgent = (agentID + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth

            # Recursive minimax call
            successor_value = self.minimax(successor, nextDepth, nextAgent)[0]

            # Update value and best action based on agent type
            if agentID == 0:  # Maximizer
                if successor_value > value:
                    value, best_action = successor_value, action
            else:  # Minimizer
                if successor_value < value:
                    value, best_action = successor_value, action

        return value, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaPruning(gameState, depth, agentID, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentID)
            if not legalActions:  # No legal actions
                return self.evaluationFunction(gameState)

            if agentID == 0:  # Maximizing agent
                value = float('-inf')
                bestAction = Directions.STOP
                for action in legalActions:
                    successorValue = alphaBetaPruning(
                        gameState.generateSuccessor(agentID, action), depth, 1, alpha, beta
                    )
                    if successorValue > value:
                        value = successorValue
                        bestAction = action
                    if value > beta:
                        return value
                    alpha = max(alpha, value)

                return bestAction if depth == 0 else value

            else:  # Minimizing agent
                value = float('inf')
                nextAgent = (agentID + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth

                for action in legalActions:
                    successorValue = alphaBetaPruning(
                        gameState.generateSuccessor(agentID, action), nextDepth, nextAgent, alpha, beta
                    )
                    value = min(value, successorValue)
                    if value < alpha:
                        return value
                    beta = min(beta, value)

                return value

        return alphaBetaPruning(gameState, 0, 0, float('-inf'), float('inf'))



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectiMax(gameState, depth, agentID):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentID)  # Cache legal actions
            if not legalActions:  # No actions available
                return self.evaluationFunction(gameState)

            if agentID == 0:  # Pacman's turn (Maximizer)
                value = float('-inf')
                bestAction = Directions.STOP
                for action in legalActions:
                    successorValue = expectiMax(gameState.generateSuccessor(agentID, action), depth, 1)
                    if successorValue > value:
                        value = successorValue
                        bestAction = action
                return bestAction if depth == 0 else value
            else:  # Ghosts' turn (Expectimax)
                nextAgent = (agentID + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth

                probability = 1 / len(legalActions)
                return sum(probability * expectiMax(gameState.generateSuccessor(agentID, action), nextDepth, nextAgent)
                           for action in legalActions)

        return expectiMax(gameState, 0, 0)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function considers the distance to the nearest food,
    the current game score, the number of remaining foods, the number of remaining capsules,
    and proximity to ghosts. It prioritizes escaping ghosts if they are too close.
    """
    "*** YOUR CODE HERE ***"

    # Setup information to be used as arguments in evaluation function
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    game_score = currentGameState.getScore()

    # Initialize closest food distance to a large value
    closest_food = float('inf') if food_count > 0 else 1

    # Combine calculation of food and ghost distances to avoid unnecessary iterations
    for food_position in food_list:
        # Update the closest food distance
        food_distance = manhattanDistance(pacman_position, food_position)
        if food_distance < closest_food:
            closest_food = food_distance

    # Check ghost proximity and prioritize escaping if necessary
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)
        if ghost_distance < 2:
            closest_food = 99999  # Override food distance to prioritize escaping
            break  # No need to check further once escape is prioritized

    features = [
        1.0 / closest_food,
        game_score,
        food_count,
        capsule_count
    ]

    weights = [
        10,
        200,
        -100,
        -10
    ]

    # Linear combination of features
    return sum(feature * weight for feature, weight in zip(features, weights))


# Abbreviation
better = betterEvaluationFunction
