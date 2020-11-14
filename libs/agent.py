from collections import deque
import random
import numpy as np
from libs.model import mlp

# DQN 의 중심적인 기능을 제공하는 Class 이다.
# 실질적인 실행은 run.py 에 있다.
# run.py: 목표신경망 update 주기, 주 신경망 학습 주기 및 시작 시점 (Observe: 처음엔 가중치가 없음),
# agent.py: target_network_update ok, memory 관리에 대하여..(remember)- deque maxlen 설정으로 괜춘


class DQNAgent(object):
    """ A simple Deep Q agent """

    # Agent 초기화 함수
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # state 개수
        self.action_size = action_size  # action 개수
        self.memory = deque(maxlen=2000)  # 각 step을 저장할 메모리
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.001
        self.model = mlp(state_size, action_size)  # Main 모델 객체
        # target model 을  설정
        self.target_model = mlp(state_size, action_size)

    # target Network update 함수
    def update_target_model():
        self.target_model.set_weights(self.model.get_weights)

    # 각 Step에 대한결과 저장함수
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 현재 State에 대한 action 도출 함수
    def act(self, state):

        # exploitation vs exploration

        # if exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # if exploration
        # 현재 신경망에 의해 Action 도출
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # 모델 학습 함수
    def replay(self, batch_size=32):
        """ vectorized implementation; 30x speed up compared with for loop """
        # 랜덤으로 메모리에서 Step에 대한 정보를 batch_size 만큼 가져오기
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a) -> for TD(target difference) Algorithm
        # model -> target_model! (목표 신경망으로 바꾸자 [])
        target = rewards + self.gamma * \
            np.argmax(self.target_model.predict(next_states), axis=1)

        # end state target is reward itself (no lookahead)
        # 마지막 state의 target은 기존 reward로 진행.. why?
        target[done] = rewards[done]

        # Q(s, a)
        # 학습을 위한 정답 Label 공정하기
        target_f = self.model.predict(states)
        # 목표 신경망을 통해 전달된 target(value)값을 알맞은 action 자리에 넣어준다.
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

    def deprecate_epsilon():
        # epsilon depreacate
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
