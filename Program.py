import csv
import os
import time
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from keras.models import load_model

import Plotter
from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
from DataLoader import DataLoader
from HumanQLearning import HumanQLearning
from Plotter import plot_actions_stats, plot_conf_matrix, print_classification_details
from ImageHelper import NumpyImg2Tensor
from QLearningModel import QLearningModel
from HumanDoubleQLearning import HumanDoubleQLearning
from StatisticsController import StatisticsController

config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4)
session = tf.compat.v1.Session(config=config)

# set to true is algorithm is launched for the first time
LOAD_DATA = False
TRAIN_NETWORK = False

TRAIN_QL = True
LIMIT = 10
ACTION_NAMES = ['rotate +90', 'rotate +180', 'diagonal translation']
networkName = "Inception"

# ----------Data Load----------------
t1 = time.time()
IMG_SIZE = 75

classes = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
dl = DataLoader("dataset/FERG_DB_256",
                ".png",
                classes,
                IMG_SIZE,
                LIMIT)

# Number of splits
num_splits = 10

# Lists to store the splits after loading saved train test split
train_splits = []
test_splits = []
validation_splits = []

if LOAD_DATA:
    X, y = dl.load()
    print('Number of samples: ' + str(len(X)))

    # Initialize the KFold cross-validator
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Create directories to store the splits if they don't exist
    os.makedirs('splits', exist_ok=True)

    # Iterate through the splits
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        # Divide the current split into training, testing, and validation sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Further splitting training data into training and validation sets
        val_fraction = 0.2
        val_index = int(val_fraction * len(X_train))

        X_validation = X_train[:val_index]
        y_validation = y_train[:val_index]

        X_train_partial = X_train[val_index:]
        y_train_partial = y_train[val_index:]

        # Save the splits as numpy arrays
        np.save(f'splits/train_split_{idx}_X.npy', X_train)
        np.save(f'splits/train_split_{idx}_y.npy', y_train)
        np.save(f'splits/test_split_{idx}_X.npy', X_test)
        np.save(f'splits/test_split_{idx}_y.npy', y_test)
        np.save(f'splits/validation_split_{idx}_X.npy', X_validation)
        np.save(f'splits/validation_split_{idx}_y.npy', y_validation)

        # Organize
        train_splits.append((X_train, y_train))
        test_splits.append((X_test, y_test))
        validation_splits.append((X_validation, y_validation))
else:
    for idx in range(num_splits):
        # Load the saved arrays into the splits arrays
        X_train = np.load(f'splits/train_split_{idx}_X.npy', allow_pickle=True)
        y_train = np.load(f'splits/train_split_{idx}_y.npy', allow_pickle=True)
        X_test = np.load(f'splits/test_split_{idx}_X.npy', allow_pickle=True)
        y_test = np.load(f'splits/test_split_{idx}_y.npy', allow_pickle=True)
        X_validation = np.load(f'splits/validation_split_{idx}_X.npy', allow_pickle=True)
        y_validation = np.load(f'splits/validation_split_{idx}_y.npy', allow_pickle=True)

        # Organize
        train_splits.append((X_train, y_train))
        test_splits.append((X_test, y_test))
        validation_splits.append((X_validation, y_validation))

print("Data Load time: " + str(time.time() - t1))
# --------Load data done--------------

# ---------CNN training---------------
epochs = 30

inception_models = []
resNet_models = []
mobileNet_models = []

accuracy_lists_inception_model = []
val_accuracy_lists_inception_model = []
loss_lists_inception_model = []
val_loss_lists_inception_model = []

accuracy_lists_resNet_model = []
val_accuracy_lists_resNet_model = []
loss_lists_resNet_model = []
val_loss_lists_resNet_model = []

accuracy_lists_mobileNet_model = []
val_accuracy_lists_mobileNet_model = []
loss_lists_mobileNet_model = []
val_loss_lists_mobileNet_model = []

if TRAIN_NETWORK:
    print('\nBegin CNN training...')
    start = time.time()

    for i in range(0, num_splits):
        X_train = train_splits[i][0]
        y_train = train_splits[i][1]
        X_val = validation_splits[i][0]
        y_val = validation_splits[i][1]

        # -------Inception-------
        print(f'Training InceptionV3 (fold {i})...')
        inception_model = ConvolutionalNeuralNetworks('Inception')
        image_shape = X_train[0].shape
        inception_model.create_model_architecture(image_shape)
        history_inception_model = inception_model.model.fit(X_train, dl.toOneHot(y_train), epochs=epochs, batch_size=20,
                                                            validation_data=(X_val, dl.toOneHot(y_val)))
        inception_model.model.save(f'models/inception_model_fold_{i}.h5')
        inception_models.append(inception_model)

        accuracy_list_inception_model = history_inception_model.history['accuracy']
        np.save(os.path.join('accuracy_lists', f'inception_accuracy_list_split_{i}.npy'), accuracy_list_inception_model)
        accuracy_lists_inception_model.append(accuracy_list_inception_model)
        val_accuracy_list_inception_model = history_inception_model.history['val_accuracy']
        np.save(os.path.join('val_accuracy_lists', f'inception_val_accuracy_list_split_{i}.npy'), val_accuracy_list_inception_model)
        val_accuracy_lists_inception_model.append(val_accuracy_list_inception_model)

        loss_list_inception_model = history_inception_model.history['loss']
        np.save(os.path.join('loss_lists', f'inception_loss_list_split_{i}.npy'), loss_list_inception_model)
        loss_lists_inception_model.append(loss_list_inception_model)
        val_loss_list_inception_model = history_inception_model.history['val_loss']
        np.save(os.path.join('val_loss_lists', f'inception_val_loss_list_split_{i}.npy'), val_loss_list_inception_model)
        val_loss_lists_inception_model.append(val_loss_list_inception_model)

        # -------ResNet-------
        print(f'Training ResNet50 (fold {i})...')
        resNet_model = ConvolutionalNeuralNetworks('ResNet')
        image_shape = X_train[0].shape
        resNet_model.create_model_architecture(image_shape)
        history_resNet_model = resNet_model.model.fit(X_train, dl.toOneHot(y_train), epochs=epochs, batch_size=20,
                                                      validation_data=(X_val, dl.toOneHot(y_val)))
        resNet_model.model.save(f'models/resNet_model_fold_{i}.h5')
        resNet_models.append(resNet_model)

        accuracy_list_resNet_model = history_resNet_model.history['accuracy']
        np.save(os.path.join('accuracy_lists', f'resNet_accuracy_list_split_{i}.npy'), accuracy_list_resNet_model)
        accuracy_lists_resNet_model.append(accuracy_list_resNet_model)
        val_accuracy_list_resNet_model = history_resNet_model.history['val_accuracy']
        np.save(os.path.join('val_accuracy_lists', f'resNet_val_accuracy_list_split_{i}.npy'), val_accuracy_list_resNet_model)
        val_accuracy_lists_resNet_model.append(val_accuracy_list_resNet_model)

        loss_list_resNet_model = history_resNet_model.history['loss']
        np.save(os.path.join('loss_lists', f'resNet_loss_list_split_{i}.npy'), loss_list_resNet_model)
        loss_lists_resNet_model.append(loss_list_resNet_model)
        val_loss_list_resNet_model = history_resNet_model.history['val_loss']
        np.save(os.path.join('val_loss_lists', f'resNet_val_loss_list_split_{i}.npy'), val_loss_list_resNet_model)
        val_loss_lists_resNet_model.append(val_loss_list_resNet_model)

        # -------MobileNet-------
        print(f'Training MobileNetV2 (fold {i})...')
        mobileNet_model = ConvolutionalNeuralNetworks('MobileNet')
        image_shape = X_train[0].shape
        mobileNet_model.create_model_architecture(image_shape)
        history_mobileNet_model = mobileNet_model.model.fit(X_train, dl.toOneHot(y_train), epochs=epochs, batch_size=20,
                                                      validation_data=(X_val, dl.toOneHot(y_val)))
        mobileNet_model.model.save(f'models/mobileNet_model_fold_{i}.h5')
        mobileNet_models.append(mobileNet_model)

        accuracy_list_mobileNet_model = history_mobileNet_model.history['accuracy']
        np.save(os.path.join('accuracy_lists', f'mobileNet_accuracy_list_split_{i}.npy'), accuracy_list_mobileNet_model)
        accuracy_lists_mobileNet_model.append(accuracy_list_mobileNet_model)

        val_accuracy_list_mobileNet_model = history_mobileNet_model.history['val_accuracy']
        np.save(os.path.join('val_accuracy_lists', f'mobileNet_val_accuracy_list_split_{i}.npy'), val_accuracy_list_mobileNet_model)
        val_accuracy_lists_mobileNet_model.append(val_accuracy_list_mobileNet_model)

        loss_list_mobileNet_model = history_mobileNet_model.history['loss']
        np.save(os.path.join('loss_lists', f'mobileNet_loss_list_split_{i}.npy'), loss_list_mobileNet_model)
        loss_lists_mobileNet_model.append(loss_list_mobileNet_model)

        val_loss_list_mobileNet_model = history_mobileNet_model.history['val_loss']
        np.save(os.path.join('val_loss_lists', f'mobileNet_val_loss_list_split_{i}.npy'), val_loss_list_mobileNet_model)
        val_loss_lists_mobileNet_model.append(val_loss_list_mobileNet_model)

    end = time.time()
    training_time = np.round((end - start) / 60, 2)
    print('CNN training completed.')
    print('CNN training time: ' + str(training_time) + 'min(s)' + '.\n')
else:
    for i in range(0, num_splits):
        accuracy_list_inception_model = np.load(os.path.join('accuracy_lists', f'inception_accuracy_list_split_{i}.npy'))
        val_accuracy_list_inception_model = np.load(os.path.join('val_accuracy_lists', f'inception_val_accuracy_list_split_{i}.npy'))
        loss_list_inception_model = np.load(os.path.join('loss_lists', f'inception_loss_list_split_{i}.npy'))
        val_loss_list_inception_model = np.load(os.path.join('val_loss_lists', f'inception_val_loss_list_split_{i}.npy'))

        accuracy_list_resNet_model = np.load(os.path.join('accuracy_lists', f'resNet_accuracy_list_split_{i}.npy'))
        val_accuracy_list_resNet_model = np.load(os.path.join('val_accuracy_lists', f'resNet_val_accuracy_list_split_{i}.npy'))
        loss_list_resNet_model = np.load(os.path.join('loss_lists', f'resNet_loss_list_split_{i}.npy'))
        val_loss_list_resNet_model = np.load(os.path.join('val_loss_lists', f'resNet_val_loss_list_split_{i}.npy'))

        accuracy_list_mobileNet_model = np.load(os.path.join('accuracy_lists', f'mobileNet_accuracy_list_split_{i}.npy'))
        val_accuracy_list_mobileNet_model = np.load(os.path.join('val_accuracy_lists', f'mobileNet_val_accuracy_list_split_{i}.npy'))
        loss_list_mobileNet_model = np.load(os.path.join('loss_lists', f'mobileNet_loss_list_split_{i}.npy'))
        val_loss_list_mobileNet_model = np.load(os.path.join('val_loss_lists', f'mobileNet_val_loss_list_split_{i}.npy'))

        accuracy_lists_inception_model.append(accuracy_list_inception_model)
        val_accuracy_lists_inception_model.append(val_accuracy_list_inception_model)
        loss_lists_inception_model.append(loss_list_inception_model)
        val_loss_lists_inception_model.append(val_loss_list_inception_model)
        accuracy_lists_resNet_model.append(accuracy_list_resNet_model)
        val_accuracy_lists_resNet_model.append(val_accuracy_list_resNet_model)
        loss_lists_resNet_model.append(loss_list_resNet_model)
        val_loss_lists_resNet_model.append(val_loss_list_resNet_model)
        accuracy_lists_mobileNet_model.append(accuracy_list_mobileNet_model)
        val_accuracy_lists_mobileNet_model.append(val_accuracy_list_mobileNet_model)
        loss_lists_mobileNet_model.append(loss_list_mobileNet_model)
        val_loss_lists_mobileNet_model.append(val_loss_list_mobileNet_model)

        inception_model = load_model(f'models/inception_model_fold_{i}.h5')
        inception_models.append(inception_model)

        resNet_model = load_model(f'models/resNet_model_fold_{i}.h5')
        resNet_models.append(resNet_model)

        mobileNet_model = load_model(f'models/mobileNet_model_fold_{i}.h5')
        mobileNet_models.append(mobileNet_model)

# --------CNN training done------------

# --------Plot cnn histories-----------
test_loss_list_inception = []
test_accuracy_list_inception = []
test_loss_list_resNet = []
test_accuracy_list_resNet = []
test_loss_list_mobileNet = []
test_accuracy_list_mobileNet = []

for i in range(num_splits):
    test_split = test_splits[i]
    X_test = test_split[0]
    y_test = test_split[1]

    results_inception = inception_models[i].evaluate(X_test, dl.toOneHot(y_test))
    test_loss_inception = results_inception[0]
    test_accuracy_inception = results_inception[1]
    test_loss_list_inception.append(test_loss_inception)
    test_accuracy_list_inception.append(test_accuracy_inception)

    results_resNet = resNet_models[i].evaluate(X_test, dl.toOneHot(y_test))
    test_loss_resNet = results_resNet[0]
    test_accuracy_resNet = results_resNet[1]
    test_loss_list_resNet.append(test_loss_resNet)
    test_accuracy_list_resNet.append(test_accuracy_resNet)

    results_mobileNet = mobileNet_models[i].evaluate(X_test, dl.toOneHot(y_test))
    test_loss_mobileNet = results_mobileNet[0]
    test_accuracy_mobileNet = results_mobileNet[1]
    test_loss_list_mobileNet.append(test_loss_mobileNet)
    test_accuracy_list_mobileNet.append(test_accuracy_mobileNet)

print('Final Accuracy: ')
print(f'InceptionV3: {np.mean(test_accuracy_list_inception)} ± {np.std(test_accuracy_list_inception)}')
print(f'ResNet50: {np.mean(test_accuracy_list_resNet)} ± {np.std(test_accuracy_list_resNet)}')
print(f'MobileNetV2: {np.mean(test_accuracy_list_mobileNet)} ± {np.std(test_accuracy_list_mobileNet)}')

print('\nFinal Loss: ')
print(f'InceptionV3: {np.mean(test_loss_list_inception)} ± {np.std(test_loss_list_inception)}')
print(f'ResNet50: {np.mean(test_loss_list_resNet)} ± {np.std(test_loss_list_resNet)}')
print(f'MobileNetV2: {np.mean(test_loss_list_mobileNet)} ± {np.std(test_loss_list_mobileNet)}')

# Accuracy
# Inception (Plot 0,0)
average_accuracy_list_inception_model = np.mean(accuracy_lists_inception_model, axis=0)
average_val_accuracy_list_inception_model = np.mean(val_accuracy_lists_inception_model, axis=0)
std_deviation_accuracy_list_inception_model = np.std(accuracy_lists_inception_model, axis=0)
std_deviation_val_accuracy_list_inception_model = np.std(val_accuracy_lists_inception_model, axis=0)

# ResNet (Plot 0,1)
average_accuracy_list_resNet_model = np.mean(accuracy_lists_resNet_model, axis=0)
average_val_accuracy_list_resNet_model = np.mean(val_accuracy_lists_resNet_model, axis=0)
std_deviation_accuracy_list_resNet_model = np.std(accuracy_lists_resNet_model, axis=0)
std_deviation_val_accuracy_list_resNet_model = np.std(val_accuracy_lists_resNet_model, axis=0)

# MobileNet (Plot 0,2)
average_accuracy_list_mobileNet_model = np.mean(accuracy_lists_mobileNet_model, axis=0)
average_val_accuracy_list_mobileNet_model = np.mean(val_accuracy_lists_mobileNet_model, axis=0)
std_deviation_accuracy_list_mobileNet_model = np.std(accuracy_lists_mobileNet_model, axis=0)
std_deviation_val_accuracy_list_mobileNet_model = np.std(val_accuracy_lists_mobileNet_model, axis=0)

# Loss
# Inception (Plot 1,0)
average_loss_list_inception_model = np.mean(loss_lists_inception_model, axis=0)
average_val_loss_list_inception_model = np.mean(val_loss_lists_inception_model, axis=0)
std_deviation_loss_list_inception_model = np.std(loss_lists_inception_model, axis=0)
std_deviation_val_loss_list_inception_model = np.std(val_loss_lists_inception_model, axis=0)

# ResNet (Plot 1,1)
average_loss_list_resNet_model = np.mean(loss_lists_resNet_model, axis=0)
average_val_loss_list_resNet_model = np.mean(val_loss_lists_resNet_model, axis=0)
std_deviation_loss_list_resNet_model = np.std(loss_lists_resNet_model, axis=0)
std_deviation_val_loss_list_resNet_model = np.std(val_loss_lists_resNet_model, axis=0)

# MobileNet (1,2)
average_loss_list_mobileNet_model = np.mean(loss_lists_mobileNet_model, axis=0)
average_val_loss_list_mobileNet_model = np.mean(val_loss_lists_mobileNet_model, axis=0)
std_deviation_loss_list_mobileNet_model = np.std(loss_lists_mobileNet_model, axis=0)
std_deviation_val_loss_list_mobileNet_model = np.std(val_loss_lists_mobileNet_model, axis=0)

Plotter.plot_model_history(average_accuracy_list_inception_model, average_val_accuracy_list_inception_model,
                           std_deviation_accuracy_list_inception_model, std_deviation_val_accuracy_list_inception_model,
                            average_accuracy_list_resNet_model, average_val_accuracy_list_resNet_model,
                            std_deviation_accuracy_list_resNet_model, std_deviation_val_accuracy_list_resNet_model,
                            average_accuracy_list_mobileNet_model, average_val_accuracy_list_mobileNet_model,
                            std_deviation_accuracy_list_mobileNet_model, std_deviation_val_accuracy_list_mobileNet_model,
                            average_loss_list_inception_model, average_val_loss_list_inception_model,
                            std_deviation_loss_list_inception_model, std_deviation_val_loss_list_inception_model,
                            average_loss_list_resNet_model, average_val_loss_list_resNet_model,
                            std_deviation_loss_list_resNet_model, std_deviation_val_loss_list_resNet_model,
                            average_loss_list_mobileNet_model, average_val_loss_list_mobileNet_model,
                            std_deviation_loss_list_mobileNet_model, std_deviation_val_loss_list_mobileNet_model)



# ------------------------------------------------------------------------------------------------------------------------------------------------------



# ----------RL execution--------------
inception_model_object = ConvolutionalNeuralNetworks('Inception')
X_train = train_splits[0][0]
image_shape = X_train[0].shape
inception_model_object.create_model_architecture(image_shape)

TS_QL = QLearningModel()
TS_QL_HF = HumanQLearning()
TS_DQL_HF = HumanDoubleQLearning()

def train_QL(QL_network, QL_network_name, action_selection_strategy):
    max_Q_values_lists = []
    cum_r_lists = []

    for i in range(2): # Here-1
        test_split = test_splits[i]
        X_test = test_split[0]
        y_test = test_split[1]

        for img, label in zip(X_test, y_test):
            print(f'Prediction using model: {i}')

            # Prediction before applying action
            probabilities_vector_before_applying_action = inception_models[i].predict(NumpyImg2Tensor(img))
            predictedLabel_before_applying_action = np.argmax(probabilities_vector_before_applying_action)

            if (predictedLabel_before_applying_action != label):
                QL_network.perform_iterative_Q_learning(inception_model_object, img, classes, action_selection_strategy)
                optimal_action = QL_network.choose_optimal_action()
                corrected_img = QL_network.apply_action(optimal_action, img)
                probabilities_vector_after_applying_action = inception_models[i].predict(NumpyImg2Tensor(corrected_img))
                predictedLabel_after_applying_action = np.argmax(probabilities_vector_after_applying_action)

        # Output of one split [1,2,3,4,5]
        max_Q_values_TS_QL_list = QL_network.max_q_estimates_all_images[0]
        cum_r_TS_QL_list = QL_network.cum_rewards_all_images[0]

        max_Q_values_lists.append(max_Q_values_TS_QL_list)
        cum_r_lists.append(cum_r_TS_QL_list)

    average_max_Q_values_list = np.mean(max_Q_values_lists, axis=0)
    print(f'Average max Q values {QL_network_name} list: {average_max_Q_values_list}')
    std_dev_max_Q_values_list = np.std(max_Q_values_lists, axis=0)
    print(f'Std dev max Q values {QL_network_name} list: {std_dev_max_Q_values_list}')

    average_cum_r_list = np.mean(cum_r_lists, axis=0)
    print(f'Average cumulative reward {QL_network_name} list: {average_cum_r_list}')
    std_dev_cum_r_list = np.std(cum_r_lists, axis=0)
    print(f'Std dev cumulative reward {QL_network_name} list: {std_dev_cum_r_list}')

    return average_max_Q_values_list, std_dev_max_Q_values_list, average_cum_r_list, std_dev_cum_r_list

# Arrays to save in RL (For plot 1)
average_max_Q_values_TS_QL_list = []
std_dev_max_Q_values_TS_QL_list = []
average_max_Q_values_TS_QL_HF_list = []
std_dev_max_Q_values_TS_QL_HF_list = []
average_max_Q_values_TS_DQL_HF_list = []
std_dev_max_Q_values_TS_DQL_HF_list = []

# Arrays to save in RL (For plot 1 & 2)
average_cum_r_TS_QL_list = []
std_dev_cum_r_TS_QL_list = []
average_cum_r_TS_QL_HF_list = []
std_dev_cum_r_TS_QL_HF_list = []
average_cum_r_TS_DQL_HF_list = []
std_dev_cum_r_TS_DQL_HF_list = []

random_strategy = 'random'
harmonic_decay_strategy = 'harmonic-sequence-e-decay'
one_shot_decay_strategy = 'one-shot-e-decay'

# For first plot
average_max_Q_values_TS_QL_list_random, std_dev_max_Q_values_TS_QL_list_random, average_cum_r_TS_QL_list_random, std_dev_cum_r_TS_QL_list_random = train_QL(TS_QL, 'TS-QL', random_strategy)
average_max_Q_values_TS_QL_list_harmonic, std_dev_max_Q_values_TS_QL_list_harmonic, average_cum_r_TS_QL_list_harmonic, std_dev_cum_r_TS_QL_list_harmonic = train_QL(TS_QL, 'TS-QL', harmonic_decay_strategy)
average_max_Q_values_TS_QL_list_one_shot, std_dev_max_Q_values_TS_QL_list_one_shot, average_cum_r_TS_QL_list_one_shot, std_dev_cum_r_TS_QL_list_one_shot = train_QL(TS_QL, 'TS-QL', one_shot_decay_strategy)
average_max_Q_values_TS_QL_HF_list_random, std_dev_max_Q_values_TS_QL_HF_list_random, average_cum_r_TS_QL_HF_list_random, std_dev_cum_r_TS_QL_HF_list_random = train_QL(TS_QL_HF, 'TS-QL-HF', random_strategy)
average_max_Q_values_TS_QL_HF_list_harmonic, std_dev_max_Q_values_TS_QL_HF_list_harmonic, average_cum_r_TS_QL_HF_list_harmonic, std_dev_cum_r_TS_QL_HF_list_harmonic = train_QL(TS_QL_HF, 'TS-QL-HF', harmonic_decay_strategy)
average_max_Q_values_TS_QL_HF_list_one_shot, std_dev_max_Q_values_TS_QL_HF_list_one_shot, average_cum_r_TS_QL_HF_list_one_shot, std_dev_cum_r_TS_QL_HF_list_one_shot = train_QL(TS_QL_HF, 'TS-QL-HF', one_shot_decay_strategy)

print(f'average_max_Q_values_TS_QL_list_random: {average_max_Q_values_TS_QL_list_random}')
print(f'std_dev_max_Q_values_TS_QL_list_random: {std_dev_max_Q_values_TS_QL_list_random}')
print(f'average_max_Q_values_TS_QL_list_harmonic: {average_max_Q_values_TS_QL_list_harmonic}')
print(f'std_dev_max_Q_values_TS_QL_list_harmonic: {std_dev_max_Q_values_TS_QL_list_harmonic}')
print(f'average_max_Q_values_TS_QL_list_one_shot: {average_max_Q_values_TS_QL_list_one_shot}')
print(f'std_dev_max_Q_values_TS_QL_list_one_shot: {std_dev_max_Q_values_TS_QL_list_one_shot}')

# Plot first plot
Plotter.plotActionSelectionAnalysis(range(1,21),
                                    average_max_Q_values_TS_QL_list_random, std_dev_max_Q_values_TS_QL_list_random,
                                    average_max_Q_values_TS_QL_list_harmonic, std_dev_max_Q_values_TS_QL_list_harmonic,
                                    average_max_Q_values_TS_QL_list_one_shot, std_dev_max_Q_values_TS_QL_list_one_shot,
                                    average_max_Q_values_TS_QL_HF_list_random, std_dev_max_Q_values_TS_QL_HF_list_random,
                                    average_max_Q_values_TS_QL_HF_list_harmonic, std_dev_max_Q_values_TS_QL_HF_list_harmonic,
                                    average_max_Q_values_TS_QL_HF_list_one_shot, std_dev_max_Q_values_TS_QL_HF_list_one_shot)

# For second plot
average_max_Q_values_TS_DQL_HF_list_one_shot, std_dev_max_Q_values_TS_DQL_HF_list_one_shot, average_cum_r_TS_DQL_HF_list_one_shot, std_dev_cum_r_TS_DQL_HF_list_one_shot = train_QL(TS_DQL_HF, 'TS-DQL-HF', one_shot_decay_strategy)


print(f'average_max_Q_values_TS_QL_list: {average_max_Q_values_TS_QL_list}')
print(f'std_dev_max_Q_values_TS_QL_list: {std_dev_max_Q_values_TS_QL_list}')
print(f'average_cum_r_TS_QL_list: {average_cum_r_TS_QL_list}')
print(f'std_dev_cum_r_TS_QL_list: {std_dev_cum_r_TS_QL_list}')

print(f'average_max_Q_values_TS_QL_HF_list: {average_max_Q_values_TS_QL_HF_list}')
print(f'std_dev_max_Q_values_TS_QL_HF_list: {std_dev_max_Q_values_TS_QL_HF_list}')
print(f'average_cum_r_TS_QL_HF_list: {average_cum_r_TS_QL_HF_list}')
print(f'std_dev_cum_r_TS_QL_HF_list: {std_dev_cum_r_TS_QL_HF_list}')

print(f'average_max_Q_values_TS_DQL_HF_list: {average_max_Q_values_TS_DQL_HF_list}')
print(f'std_dev_max_Q_values_TS_DQL_HF_list: {std_dev_max_Q_values_TS_DQL_HF_list}')
print(f'average_cum_r_TS_DQL_HF_list: {average_cum_r_TS_DQL_HF_list}')
print(f'std_dev_cum_r_TS_DQL_HF_list: {std_dev_cum_r_TS_DQL_HF_list}')

Plotter.plot_QL_history(range(1, 21),
                        average_max_Q_values_TS_QL_list, std_dev_max_Q_values_TS_QL_list, average_cum_r_TS_QL_list, std_dev_cum_r_TS_QL_list,
                        average_max_Q_values_TS_QL_HF_list, std_dev_max_Q_values_TS_QL_HF_list, average_cum_r_TS_QL_HF_list, std_dev_cum_r_TS_QL_HF_list,
                        average_max_Q_values_TS_DQL_HF_list, std_dev_max_Q_values_TS_DQL_HF_list, average_cum_r_TS_DQL_HF_list, std_dev_cum_r_TS_DQL_HF_list)

# ----------RL execution--------------
t4 = time.time()
TS_QL = QLearningModel()
TS_QL_HF = HumanQLearning()
TS_DQL_HF = HumanDoubleQLearning()
statControllerRl = StatisticsController(classes, len(ACTION_NAMES))
verbose = True

# Plot Learning Curve
misclassified_num = 0

with open('q_value_per_episode.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    header = ['episodes', 'q-values']

    # write the header
    writer.writerow(header)

for img, label in zip(X_test, y_test):
    no_lr_probabilities_vector = cnn.model.predict(NumpyImg2Tensor(img))
    predictedLabel = np.argmax(no_lr_probabilities_vector)
    statControllerNoRl.predictedLabels.append(predictedLabel)

    # Interactive QL
    if predictedLabel != label:
        misclassified_num += 1

        TS_QL_HF.interactive_q_learning_human_only(cnn, img, statControllerRl, classes)
        optimal_action = TS_QL_HF.choose_optimal_action()
        statControllerRl.updateOptimalActionsStats(optimal_action)
        corrected_img = TS_QL_HF.apply_action(optimal_action, img)
        probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
        statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))

        TS_DQL_HF.interactive_double_q_learning_human_only(cnn, img, statControllerRl, classes)
        optimal_action = TS_DQL_HF.choose_optimal_action()
        statControllerRl.updateOptimalActionsStats(optimal_action)
        corrected_img = TS_DQL_HF.apply_action(optimal_action, img)
        probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
        statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))

        TS_QL.perform_iterative_Q_learning(cnn, img, statControllerRl)
        optimal_action = TS_QL.choose_optimal_action()
        statControllerRl.updateOptimalActionsStats(optimal_action)
        corrected_img = TS_QL.apply_action(optimal_action, img)
        probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
        statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))
    else:
        statControllerRl.predictedLabels.append(predictedLabel)

# 1. Plot CNN history
plot_history(dl, networkName, statControllerNoRl.trainingHistory)

# 2. Plot QL learning curve
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))

q_values_all_images_ts_ql_hf = TS_QL_HF.q_estimates_all_images
q_values_all_images_ts_dql_hf = TS_DQL_HF.q_estimates_all_images
q_values_all_images_ts_ql = TS_QL.q_estimates_all_images

cum_rewards_all_images_ts_ql_hf = TS_QL_HF.cum_rewards_all_images
cum_rewards_all_images_ts_dql_hf = TS_DQL_HF.cum_rewards_all_images
cum_rewards_all_images_ts_ql = TS_QL.cum_rewards_all_images
episodes = range(1, TS_QL.maxIter+1)

# Plot Q-values
axes[0, 0].set_title('(a) Image 1')
axes[0, 0].set_xlabel('Episodes')
axes[0, 0].set_ylabel('Average Q-values')
axes[0, 0].plot(episodes, q_values_all_images_ts_dql_hf[0], color='red', label='TS-DQL-HF')
axes[0, 0].plot(episodes, q_values_all_images_ts_ql_hf[0], linestyle='dashed', color='blue', label='TS-QL-HF')
# axes[0, 0].plot(episodes, q_values_all_images_ts_ql[0], color='blue', label='TS-QL')

axes[0, 1].set_title('(b) Image 2')
axes[0, 1].set_xlabel('Episodes')
axes[0, 1].set_ylabel('Average Q-values')
axes[0, 1].plot(episodes, q_values_all_images_ts_dql_hf[1], color='red', label='TS-DQL-HF')
axes[0, 1].plot(episodes, q_values_all_images_ts_ql_hf[1], linestyle='dashed', color='blue', label='TS-QL-HF')
# axes[0, 1].plot(episodes, q_values_all_images_ts_ql[1], color='blue', label='TS-QL')

axes[0, 2].set_title('(c) Image 3')
axes[0, 2].set_xlabel('Episodes')
axes[0, 2].set_ylabel('Average Q-values')
axes[0, 2].plot(episodes, q_values_all_images_ts_dql_hf[2], color='red', label='TS-DQL-HF')
axes[0, 2].plot(episodes, q_values_all_images_ts_ql_hf[2], linestyle='dashed', color='blue', label='TS-QL-HF')
# axes[0, 2].plot(episodes, q_values_all_images_ts_ql[2], color='blue', label='TS-QL')

axes[1, 0].set_title('(d) Image 1')
axes[1, 0].set_xlabel('Episodes')
axes[1, 0].set_ylabel('Cumulative Rewards')
axes[1, 0].plot(episodes, cum_rewards_all_images_ts_dql_hf[0], color='red', label='TS-DQL-HF')
axes[1, 0].plot(episodes, cum_rewards_all_images_ts_ql_hf[0], linestyle='dashed', color='blue', label='TS-QL-HF')
# axes[1, 0].plot(episodes, cum_rewards_all_images_ts_ql[0], color='blue', label='TS-QL')

axes[1, 1].set_title('(e) Image 2')
axes[1, 1].set_xlabel('Episodes')
axes[1, 1].set_ylabel('Cumulative Rewards')
axes[1, 1].plot(episodes, cum_rewards_all_images_ts_dql_hf[1], color='red', label='TS-DQL-HF')
axes[1, 1].plot(episodes, cum_rewards_all_images_ts_ql_hf[1], linestyle='dashed', color='blue', label='TS-QL-HF')
# axes[1, 1].plot(episodes, cum_rewards_all_images_ts_ql[1], color='blue', label='TS-QL')

axes[1, 2].set_title('(f) Image 3')
axes[1, 2].set_xlabel('Episodes')
axes[1, 2].set_ylabel('Cumulative Rewards')
axes[1, 2].plot(episodes, cum_rewards_all_images_ts_dql_hf[2], color='red', label='TS-DQL-HF')
axes[1, 2].plot(episodes, cum_rewards_all_images_ts_ql_hf[2], linestyle='dashed', color='blue', label='TS-QL-HF')
# axes[1, 2].plot(episodes, cum_rewards_all_images_ts_ql[2], color='blue', label='TS-QL')

plt.figlegend(['TS-DQL-HF', 'TS-QL-HF'], loc='lower center')
plt.tight_layout()
plt.show()

print("RL execution time: " + str(time.time() - t4))

# 2. Print Action Stats
plot_actions_stats(dl, networkName, ACTION_NAMES, statControllerRl.allActionsStats, "allActionsRL")
plot_actions_stats(dl, networkName, ACTION_NAMES, statControllerRl.optimalActionsStats, "optimalActionsRL")

# 3. Print Confusion Matrix
conf_matrix_no_RL = confusion_matrix(y_test, statControllerNoRl.predictedLabels)
conf_matrix_RL = confusion_matrix(y_test, statControllerRl.predictedLabels)
plot_conf_matrix(dl, networkName, conf_matrix_no_RL, classes, "NoRL")
plot_conf_matrix(dl, networkName, conf_matrix_RL, classes, "RL")

statControllerNoRl.f1Score = f1_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.precision = precision_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.recall = recall_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.report = classification_report(y_test, statControllerNoRl.predictedLabels)
statControllerNoRl.accuracy = accuracy_score(y_test, statControllerNoRl.predictedLabels)

statControllerRl.f1Score = f1_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.precision = precision_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.recall = recall_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.report = classification_report(y_test, statControllerRl.predictedLabels)
statControllerRl.accuracy = accuracy_score(y_test, statControllerRl.predictedLabels)

# 4. Print Report (f1Score, precision, recall, accuracy)
print_classification_details(statControllerNoRl)
print_classification_details(statControllerRl)
dl.save_details(statControllerNoRl, networkName, "NoRL")
dl.save_details(statControllerRl, networkName, "RL")
