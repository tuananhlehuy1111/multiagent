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
            Tính khoảng cách đến thức ăn gần nhất.
            """
            foodList = newFood.asList()  # Lấy danh sách thức ăn
            if not foodList:  # Nếu không còn thức ăn
                return 0
            # Tính khoảng cách ngắn nhất đến thức ăn
            return min(manhattanDistance(newPos, food) for food in foodList)

        def distanceFromGhosts():
            """
            Tính điểm phạt dựa trên khoảng cách đến các ma.
            """
            penalty = 0
            # Duyệt qua tất cả các ma
            for i, ghostState in enumerate(newGhostStates):
                ghostPos = ghostState.getPosition()  # Vị trí của ma
                dist = manhattanDistance(newPos, ghostPos)  # Tính khoảng cách giữa Pacman và ma
                if newScaredTimes[i] == 0:  # Nếu ma không sợ
                    # Nếu gần ma, cộng điểm phạt lớn
                    penalty += max(0, 4 - dist) * 100
            return penalty

        if successorGameState.isWin():
            return 99999  # Điểm cao khi thắng
        if successorGameState.isLose():
            return -99999  # Điểm thấp khi thua

        # Tính khoảng cách đến thức ăn và điểm phạt từ ma
        foodDistance = distanceToNearestFood()  # Khoảng cách đến thức ăn gần nhất
        ghostPenalty = distanceFromGhosts()  # Điểm phạt từ các ma

        # Nếu dừng lại, sẽ bị phạt thêm 10 điểm
        stopPenalty = 10 if action == Directions.STOP else 0

        # Trả về tổng điểm: điểm hiện tại - phạt từ ma - phạt khi dừng lại + thưởng khi gần thức ăn
        return (
                successorGameState.getScore()  # Điểm hiện tại
                - ghostPenalty  # Trừ điểm phạt từ ma
                - stopPenalty  # Trừ điểm phạt khi dừng lại
                + (1 / (foodDistance + 1)) * 10  # Cộng điểm nếu gần thức ăn
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
        # Gọi hàm minimax thống nhất và lấy hành động tốt nhất
        return self.minimax(gameState, 0, 0)[1]

    def minimax(self, gameState, depth, agentID):
        """
        Hàm minimax thống nhất xử lý cả agent tối đa hóa (max) và tối thiểu hóa (min).
        Trả về một tuple (score, action).
        """
        # Kiểm tra trạng thái cuối cùng (game thắng/thua) hoặc độ sâu tối đa
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentID)  # Lấy danh sách hành động hợp lệ của agent
        if not legalActions:  # Nếu không còn hành động hợp lệ
            return self.evaluationFunction(gameState), None

        # Khởi tạo giá trị ban đầu cho agent tối đa hóa (Pacman) hoặc tối thiểu hóa (Ghosts)
        if agentID == 0:  # Lượt của agent tối đa hóa (Pacman)
            value, best_action = float(
                '-inf'), Directions.STOP  # Khởi tạo giá trị là âm vô cùng (tìm giá trị lớn nhất)
        else:  # Lượt của agent tối thiểu hóa (Ghosts)
            value, best_action = float('inf'), None  # Khởi tạo giá trị là dương vô cùng (tìm giá trị nhỏ nhất)

        # Lặp qua tất cả các hành động hợp lệ của agent
        for action in legalActions:
            successor = gameState.generateSuccessor(agentID,
                                                    action)  # Tạo trạng thái tiếp theo sau khi thực hiện hành động

            # Xác định agent tiếp theo và độ sâu tiếp theo
            nextAgent = (agentID + 1) % gameState.getNumAgents()  # Tính agent tiếp theo
            nextDepth = depth + 1 if nextAgent == 0 else depth  # Tăng độ sâu khi quay lại lượt của agent tối đa hóa (Pacman)

            # Gọi đệ quy minimax để tính giá trị của trạng thái tiếp theo
            successor_value = self.minimax(successor, nextDepth, nextAgent)[
                0]  # Lấy giá trị từ trạng thái tiếp theo

            # Cập nhật giá trị và hành động tốt nhất dựa trên kiểu agent (tối đa hóa hoặc tối thiểu hóa)
            if agentID == 0:  # Lượt của agent tối đa hóa (Pacman)
                if successor_value > value:
                    value, best_action = successor_value, action  # Cập nhật giá trị và hành động tốt nhất nếu tìm được giá trị lớn hơn
            else:  # Lượt của agent tối thiểu hóa (Ghosts)
                if successor_value < value:
                    value, best_action = successor_value, action  # Cập nhật giá trị và hành động tốt nhất nếu tìm được giá trị nhỏ hơn

        return value, best_action  # Trả về giá trị và hành động tốt nhất


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
            # Dừng lại khi đạt độ sâu tối đa hoặc game kết thúc (thắng/thua)
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentID)  # Lấy danh sách hành động hợp lệ của agent
            if not legalActions:  # Nếu không còn hành động hợp lệ
                return self.evaluationFunction(gameState)

            if agentID == 0:  # Lượt của agent tối đa hóa (Maximizing agent)
                value = float('-inf')  # Khởi tạo giá trị là âm vô cùng (tìm giá trị lớn nhất)
                bestAction = Directions.STOP  # Khởi tạo hành động tốt nhất là dừng lại (STOP)

                # Lặp qua tất cả các hành động hợp lệ và chọn hành động có giá trị lớn nhất
                for action in legalActions:
                    # Tính giá trị của trạng thái tiếp theo sau khi thực hiện hành động
                    successorValue = alphaBetaPruning(
                        gameState.generateSuccessor(agentID, action), depth, 1, alpha, beta
                    )
                    # Cập nhật giá trị và hành động tốt nhất nếu tìm được giá trị lớn hơn
                    if successorValue > value:
                        value = successorValue
                        bestAction = action
                    # Cắt tỉa nhánh nếu giá trị của agent đã lớn hơn beta
                    if value > beta:
                        return value
                    # Cập nhật alpha với giá trị lớn nhất giữa alpha và value
                    alpha = max(alpha, value)

                # Nếu đang ở độ sâu 0 (lúc này là lượt của agent tối đa hóa), trả về hành động tốt nhất
                return bestAction if depth == 0 else value

            else:  # Lượt của agent tối thiểu hóa (Minimizing agent)
                value = float('inf')  # Khởi tạo giá trị là dương vô cùng (tìm giá trị nhỏ nhất)
                nextAgent = (agentID + 1) % gameState.getNumAgents()  # Chuyển sang agent tiếp theo
                nextDepth = depth + 1 if nextAgent == 0 else depth  # Nếu quay lại lượt của agent tối đa hóa, tăng độ sâu

                # Lặp qua tất cả các hành động hợp lệ của agent tối thiểu hóa
                for action in legalActions:
                    # Tính giá trị của trạng thái tiếp theo sau khi thực hiện hành động
                    successorValue = alphaBetaPruning(
                        gameState.generateSuccessor(agentID, action), nextDepth, nextAgent, alpha, beta
                    )
                    # Cập nhật giá trị nếu tìm được giá trị nhỏ hơn
                    value = min(value, successorValue)
                    # Cắt tỉa nhánh nếu giá trị của agent đã nhỏ hơn alpha
                    if value < alpha:
                        return value
                    # Cập nhật beta với giá trị nhỏ nhất giữa beta và value
                    beta = min(beta, value)

                return value

        # Gọi hàm alphaBetaPruning bắt đầu từ độ sâu 0 và agentID 0 (lượt của agent tối đa hóa)
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
            # Dừng lại khi đạt độ sâu tối đa hoặc game kết thúc (thắng/thua)
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentID)  # Lấy danh sách hành động hợp lệ của agent
            if not legalActions:  # Nếu không còn hành động hợp lệ
                return self.evaluationFunction(gameState)

            if agentID == 0:  # Lượt của Pacman (Maximizer)
                value = float('-inf')  # Khởi tạo giá trị của Pacman là âm vô cùng (tìm giá trị lớn nhất)
                bestAction = Directions.STOP  # Khởi tạo hành động tốt nhất là dừng lại (STOP)

                # Lặp qua tất cả các hành động hợp lệ của Pacman và chọn hành động có giá trị tốt nhất
                for action in legalActions:
                    # Tính giá trị của trạng thái tiếp theo sau khi thực hiện hành động
                    successorValue = expectiMax(gameState.generateSuccessor(agentID, action), depth, 1)
                    # Cập nhật giá trị và hành động tốt nhất nếu tìm được giá trị lớn hơn
                    if successorValue > value:
                        value = successorValue
                        bestAction = action

                # Nếu đang ở độ sâu 0 (lúc này là lượt của Pacman), trả về hành động tốt nhất
                return bestAction if depth == 0 else value

            else:  # Lượt của các ma (Expectimax)
                nextAgent = (agentID + 1) % gameState.getNumAgents()  # Chuyển sang agent tiếp theo (ma kế tiếp)
                nextDepth = depth + 1 if nextAgent == 0 else depth  # Nếu quay lại lượt của Pacman, tăng độ sâu

                # Xác suất mỗi hành động được chọn (giả định xác suất bằng nhau cho mỗi hành động)
                probability = 1 / len(legalActions)

                # Tính tổng giá trị kỳ vọng của các hành động, nhân với xác suất của mỗi hành động
                return sum(probability * expectiMax(gameState.generateSuccessor(agentID, action), nextDepth, nextAgent)
                           for action in legalActions)

        # Gọi hàm expectiMax bắt đầu từ độ sâu 0 và agentID 0 (Pacman)
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
    # Lấy thông tin cần thiết từ GameState
    newPos = currentGameState.getPacmanPosition()  # Vị trí hiện tại của Pacman
    newFood = currentGameState.getFood().asList()  # Danh sách thức ăn

    # Tính khoảng cách Manhattan đến thức ăn gần nhất
    minFoodDist = float('inf')
    for food in newFood:
        minFoodDist = min(minFoodDist, manhattanDistance(newPos, food))

    # Tính điểm phạt dựa trên khoảng cách đến các con ma
    ghostPenalty = 0
    for ghost in currentGameState.getGhostPositions():
        ghostPenalty = manhattanDistance(newPos, ghost)
        if ghostPenalty < 2:  # Nếu Pacman quá gần ma
            return -float('inf')  # Phạt nặng khi va chạm với ma

    # Lấy số lượng thức ăn và viên thuốc còn lại
    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())

    # Thiết lập các hệ số tính điểm
    foodLeftScore = 1000000  # Điểm cao khi ít thức ăn
    capsLeftScore = 10000  # Điểm cao khi còn viên thuốc
    foodDistScore = 500  # Điểm cao khi gần thức ăn

    # Thêm các yếu tố điểm thưởng/phạt khi thắng hoặc thua
    additionalScore = 0
    if currentGameState.isLose():  # Nếu thua
        additionalScore -= 100000  # Điểm phạt khi thua
    elif currentGameState.isWin():  # Nếu thắng
        additionalScore += 100000  # Điểm thưởng khi thắng

    # Tính tổng điểm để đánh giá trạng thái
    return (
            1.0 / (foodLeft + 1) * foodLeftScore  # Điểm cao hơn khi còn ít thức ăn
            + ghostPenalty  # Điểm phạt từ khoảng cách đến ma
            + 1.0 / (minFoodDist + 1) * foodDistScore  # Điểm cao hơn khi gần thức ăn
            + 1.0 / (capsLeft + 1) * capsLeftScore  # Điểm cao hơn khi còn ít viên thuốc
            + additionalScore  # Điểm thưởng/phạt khi thắng hoặc thua
    )

# Abbreviation
better = betterEvaluationFunction
