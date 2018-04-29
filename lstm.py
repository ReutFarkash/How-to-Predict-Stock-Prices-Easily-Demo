import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white',figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print('yo')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def plot_results_multiple_robust(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white',figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print('yo')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        if i % 10 == 0:
            padding = [None for p in range(i)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
    plt.show()

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result_norm, window_mean, window_std = normalise_windows(result)
    else:
        result_norm = result
        parms = ()

    result_norm = np.array(result_norm)

    row = round(0.9 * result_norm.shape[0])
    train = result_norm[:int(row), :]
    print("train.shape", train.shape)
    if normalise_window:
        train_window_mean = window_mean[:int(row)]
        train_window_std = window_std[:int(row)]
    prmut = np.arange(round(train.shape[0]))
    np.random.shuffle(prmut)
    print("prmut.shape", prmut.shape)
    train = train[prmut]
    print("train.shape", train.shape)
    if normalise_window:
        print("train_window_mean.shape", np.array(train_window_mean).shape)
        train_window_mean = train_window_mean[prmut]
        train_window_std = train_window_std[prmut]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result_norm[int(row):, :-1]
    y_test = result_norm[int(row):, -1]
    if normalise_window:
        test_window_mean = window_mean[int(row):]
        test_window_std = window_std[int(row):]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  
    if normalise_window:
        parms = (window_mean, window_std, train_window_mean, train_window_std, test_window_mean, test_window_std)

    return [x_train, y_train, x_test, y_test, parms]

def normalise_windows(window_data):
    normalised_data = []
    window_mean = []
    window_std = []
    for window in window_data:
        window = [float(i) for i in window]
        mean = np.mean((window))
        std = np.std((window))
        normalised_window = [((float(p) - mean)/std) for p in window]
        normalised_data.append(normalised_window)
        window_mean.append(mean)
        window_std.append(std)
    return normalised_data, np.array(window_mean), np.array(window_std)

def denormalise_windows(window_data, window_mean, window_std):
    denormalised_data = []
    for index, window in enumerate(window_data):
        window = [float(i) for i in window]
        mean = window_mean[index]
        std = window_std[index]
        denormalised_window = [(float(p)*std + mean) for p in window]
        denormalised_data.append(denormalised_window)
    return denormalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(len(data)//prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_sequences_multiple_robust(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 1 steps
    prediction_seqs = []
    for i in range(len(data)):
        curr_frame = data[i]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs