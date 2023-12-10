import random
import matplotlib.pyplot as plt

class Board:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        
    def display(self):
        for row in self.board:
            print('|'.join(row))
    
    def is_winner(self, player):
        # Проверка наличия победителя
        lines = self.board + list(zip(*self.board)) + [[self.board[i][i] for i in range(3)], [self.board[i][2-i] for i in range(3)]]
        return any(all(cell == player for cell in line) for line in lines)
    
    def available_moves(self):
        # Получение доступных ходов
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
    
    def make_move(self, move, player):
        # Выполнение хода
        i, j = move
        self.board[i][j] = player

class Agent:
    def __init__(self, epsilon=0.1, alpha=0.5):
        self.epsilon = epsilon  # Параметр исследования
        self.alpha = alpha  # Скорость обучения
        self.values = {}  # Стратегия агента
    
    def get_value(self, state):
        # Получение значения для данного состояния
        if state not in self.values:
            self.values[state] = 0.5  # Инициализация значения
        return self.values[state]
    
    def update_value(self, state, next_state, reward):
        # Обновление значения на основе полученной награды
        value = self.get_value(state)
        next_value = self.get_value(next_state)
        updated_value = value + self.alpha * (reward + next_value -value)
        self.values[state] = updated_value
    
    def choose_move(self, board):
        # Выбор хода на основе текущей стратегии агента
        if random.random() < self.epsilon:
            # Исследование: случайный ход
            return random.choice(board.available_moves())
        else:
            # Использование текущей стратегии
            best_move = None
            best_value = float('-inf')
            for move in board.available_moves():
                board.make_move(move, 'X')
                state = str(board.board)
                value = self.get_value(state)
                if value > best_value:
                    best_move = move
                    best_value = value
                board.make_move(move, ' ')
            return best_move

def train(agent, epochs):
    loc_rewards = 0
    rewards = []
    for epoch in range(epochs):
        board = Board()
        random_agent = Agent(epsilon=1)  # Случайный агент
        
        #print(f"Эпоха {epoch+1}/{epochs}")
        #board.display()
        
        while True:
            # Ход обучаемого агента
            state = str(board.board)
            move = agent.choose_move(board)
            board.make_move(move, 'X')
            
            if board.is_winner('X'):
                # Обучаемый агент победил
                agent.update_value(state, str(board.board), 1)
                loc_rewards += 1
                #rewards.append(1)
                print("Обучаемый агент победил!")
                break
            elif len(board.available_moves()) == 0:
                # Ничья
                agent.update_value(state, str(board.board), 0)
                loc_rewards -= 1
                #rewards.append(0)
                print("Ничья!")
                break
            
            # Ход случайного агента
            random_move = random_agent.choose_move(board)
            board.make_move(random_move, 'O')
            
            if board.is_winner('O'):
                # Случайный агент победил
                agent.update_value(state, str(board.board), -1)
                #rewards.append(-1)
                loc_rewards -= 1
                #loc_rewards = 0
                print("Случайный агент победил!")
                break
                
            #print("Ход обучаемого агента:")
            #board.display()
            #print("Ход случайного агента:")
            #board.display()
            #print("---------------------")
        rewards.append(loc_rewards)
                
    return rewards

agent = Agent()
epochs = 10000
rewards = train(agent, epochs)
print(rewards)

# Построение кривой обучения
plt.plot(range(1, epochs+1), rewards)
plt.xlabel('Количество шагов обучения')
plt.ylabel('Награда')
plt.title('Кривая обучения')
plt.show()
