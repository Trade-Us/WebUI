# dqn
import pickle
import datetime as dt
import time as t
import argparse
import re
from libs.envs import TradingEnv
from libs.agent import DQNAgent
from libs.utils import get_data, get_scaler, maybe_make_dir

# flask
import flask
from flask import Flask, request, render_template

#from sklearn.externals import joblib
import numpy as np
from scipy import misc
import imageio

# lstm
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import datetime
app = Flask(__name__)

# jinja 연동하기 위한 Bracket 설정 (jQuery)
jinja_options = app.jinja_options.copy()
jinja_options.update(dict(
    variable_start_string='((',
    variable_end_string='))'
))
app.jinja_options = jinja_options

# 메인 페이지 라우팅


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict/DQN', methods=['POST'])
def run_DQN():

    #global time
    #parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--episode', type=int, default=2000,
    # help='number of episode to run')
    # parser.add_argument('-b', '--batch_size', type=int, default=64,
    # help='batch size for experience replay')
    # parser.add_argument('-i', '--initial_invest', type=int, default=2000000,
    # help='initial investment amount')
    # parser.add_argument('-m', '--mode', type=str, required=True,
    # help='either "train" or "test"')
    #parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
    #args = parser.parse_args()
    mode = 'train'
    initial_invest = 2000000
    episode = 10
    batch_size = 64
    weights = 'path of weights file'
    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    timestamp = t.strftime('%m%d%S')

    data = np.around(get_data())
    train_data = data[:, :]
    test_data = data[:, :]

    env = TradingEnv(train_data, initial_invest)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # Limit Var
    OBSERVE = 0
    TRAIN_INTERVAL = 4
    TARGET_UPDATE_INTERVAL = 500
    time_step = 0

    # Action Collection
    # actions = np.zeros((args.episode, env.n_step))
    actions = []  # np -> list (저장수단)

    # Portfolio
    portfolio_value = []

    if mode == 'test':
        # remake the env with test data
        env = TradingEnv(test_data, initial_invest)
        # load trained weights
        agent.load(args.weights)
        # when test, the timestamp is same as time when weights was trained
        timestamp = re.findall(r'\d{6}', args.weights)[0]

    for e in range(episode):
        state = env.reset()
        state = scaler.transform([state])
        actions = []
        for time in range(env.n_step):
            # count step time
            time_step += 1

            # go step
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])

            # Collecting Actions
            # actions[e, time] = action
            actions.append(action)  # 해당 에피소드만 저장

            # remember steps
            if mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state

            # episode done
            if done:
                print(actions, end="")
                print(" ", end="")
                print("episode: {}/{}, episode end value: {}".format(
                    e + 1, episode, info['cur_val']))

                # append episode end portfolio value
                portfolio_value.append([actions, info['cur_val']])
                break

            # train Network
            # train 에 대한 Observe 및 주기 설정
            if mode == 'train' and time_step > OBSERVE:
                if len(agent.memory) > batch_size and time_step % TRAIN_INTERVAL == 0:
                    agent.replay(batch_size)

                # target Network 업데이트에 대한 주기 설정
                if time_step % TARGET_UPDATE_INTERVAL == 1:
                    agent.update_target_model

            # 입실론 감쇄에 대한 Observe 설정
            if mode == 'train' and e > OBSERVE:
                agent.deprecate_epsilon

        # Save Weights
        if mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
            agent.save(
                'weights/{}-b{}-e{}.h5'.format(timestamp, batch_size, episode))

    # save portfolio value history to disk
    with open('portfolio_val/{}-b{}-e{}-{}.p'.format(timestamp, batch_size, episode, mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)

    with open('portfolio_val/{}-b{}-e{}-{}.p'.format(timestamp, batch_size, episode, mode), 'rb') as tempfile:
        portfol_val = pickle.load(tempfile)
        # print(portfol_val)

    # Find Maximum Value and Index
    max_v = portfol_val[0][1]
    index = 0
    i = 0
    for vals in portfol_val:
        if vals[1] > max_v:
            max_v = vals[1]
            index = i
        i += 1
        #print(vals[0], vals[1])

    print("Maximum Action & Value & index")
    print(portfol_val[index][0], max_v, index)

    # dqnresult = str(portfol_val[index][0]) + " " + \
    #    str(max_v)+"원 "+str(index+1)+"번째 에피소드"

    today = dt.datetime.today()
    step = 0
    stock_num = 0
    exchange = initial_invest
    dqnresult = []
    while step < len(portfol_val[index][0]) + 1:
        action = portfol_val[index][0][step] if step < len(portfol_val[index][0]) else 1
        if action == 0:
            action = "Sell"
            exchange = public_label[step] * stock_num if stock_num > 0 else exchange
            stock_num = 0
        elif action == 1:
            action = "Hold"
        elif action == 2:
            action = "Buy"
            if exchange > public_label[step]:
                stock_num = exchange // public_label[step] 
                exchange -= stock_num * public_label[step]
        get_val = (public_label[step] * stock_num) + exchange
        profit = get_val - initial_invest
        step += 1
        date = (today + dt.timedelta(days=step)).strftime("%Y-%m-%d")
        dqnresult.append([date, action, get_val, profit])
    return render_template('index.html', labe=public_label, dqnResult=dqnresult)

# 데이터 예측 처리


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        # 업로드 파일 처리 분기
        file = request.files["trainFile"]
        if not file:
            return render_template('index.html', labe="No Files")
        file.save("./lstmweights/"+file.filename)
        data = pd.read_csv('./lstmweights/'+file.filename)  # csv파일 로드

        high_prices = data['High'].values
        low_prices = data['Low'].values
        mid_prices = (high_prices + low_prices) / 2  # midprice로 예측

        # 정규화를 위한 코드 추가
        max_price = max(mid_prices)
        min_price = min(mid_prices)

        seq_len = 100  # 며칠간의 데이터를 보고 내일것을 예측할거냐
        sequence_length = seq_len + 1  # 50개를 보고 1개를 예측 //51개 데이터를 한 window로 만듦

        # Windowing
        result = []
        for index in range(len(mid_prices) - sequence_length):
            result.append(mid_prices[index:index + sequence_length])

        # 역정규화 위한 정규화
        normalized_data = []

        for window in result:
            normalized_window = [(float(p) - min_price) /
                                 (max_price + min_price) for p in window]
            normalized_data.append(normalized_window)
        result = np.array(normalized_data)

        # 기존 정규화
        # normalized_data = []
        # for window in result:
        #     normalized_window = [((float(p) / float(window[0])) - 1)
        #                          for p in window]
        #     normalized_data.append(normalized_window)

        # result = np.array(normalized_data)

        row = int(round(result.shape[0] * 0.9))  # 전체 데이터의 90%를 트레이닝셋
        train = result[:row, :]
        np.random.shuffle(train)  # 랜덤으로 섞는다

        x_train = train[:, :-1]  # 50일간의 데이터셋으로
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        y_train = train[:, -1]  # 나머지 1일 예측

        x_test = result[row:, :-1]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test = result[row:, -1]

        # 이미지 픽셀 정보 읽기
        # 알파 채널 값 제거 후 1차원 Reshape
#         img = imageio.imread('./lstmweights/'+file.filename)
#         img = img[:, :, :3]
#         img = img.reshape(1, -1)

#         # 입력 받은 이미지 예측
#         prediction = model.predict(img)

#         # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
#         label = str(np.squeeze(prediction))
        origin_seq_in = np.array(x_test)
        seq_in = origin_seq_in[0]
        seq_count = 0

        seq_out = seq_in
        pred = np.zeros((100, 1))
        for i in range(60):
            seq_count = seq_count + 1
            if seq_count % 10 == 0:

                seq_in = origin_seq_in[seq_count]
            sample_in = np.array(seq_in)
            sample_in = np.reshape(sample_in, (1, 100, 1))

            pred_out = model.predict(sample_in)

            # print(pred_out)
            seq_in = np.append(seq_in, pred_out, axis=0)

            # seq_in.append(pred_out[0,0])
            # seq_in.pop(0)
            seq_in = np.delete(seq_in, [0, 0], axis=0)
            pred[i, 0] = pred_out[0, 0]

        # 역정규화
        counter_normalize = (pred * (max_price + min_price)) + min_price

        # 숫자가 10일 경우 0으로 처리
        #if label == '10': label = '0'
        for i in range(0, 10):
            public_label.append(counter_normalize[i, 0])
        # 결과 리턴
        return render_template('index.html', labe=public_label)


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = Sequential()  # 모델을 순차적으로 정의하는 클래스

    model.add(LSTM(100, return_sequences=True, input_shape=(100, 1)))

    model.add(LSTM(64, return_sequences=False))
 #           ------ 조정하면 서 성능테스트
    model.add(Dense(1, activation='linear'))
#              ---output 개수: 다음날 하루의 output
    model.compile(loss='mse', optimizer='rmsprop')
    model.load_weights('lstmweights/202011072326-lstm.h5')

    public_label = []
    # Flask 서비스 스타트
    app.run(host='127.0.0.1', port=8000, debug=True)
