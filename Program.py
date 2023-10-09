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

TRAIN_QL = False
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

    print(inception_models[0].summary())
    print(resNet_models[0].summary())
    print(mobileNet_models[0].summary())

    total_params = sum(p.numel() for p in inception_models[0].trainable_variables)
    print("Total number of neurons in InceptionV3:", total_params)

    total_params = sum(p.numel() for p in inception_models[0].trainable_variables)
    print("Total number of neurons in ResNet50:", total_params)

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

# --------Plot cnn final f1 score-----------
f1_scores_inception = []
f1_scores_resNet = []
f1_scores_mobileNet = []

for i in range(num_splits):
    test_split = test_splits[i]
    X_test = test_split[0]
    y_test = test_split[1]

    predicted_labels_inception = []
    predicted_labels_resNet = []
    predicted_labels_mobileNet = []

    for img, label in zip(X_test, y_test):
        probabilities_vector_inception = inception_models[i].predict(NumpyImg2Tensor(img))
        predictedLabel_inception = np.argmax(probabilities_vector_inception)
        predicted_labels_inception.append(predictedLabel_inception)

        probabilities_vector_resNet = resNet_models[i].predict(NumpyImg2Tensor(img))
        predictedLabel_resNet = np.argmax(probabilities_vector_resNet)
        predicted_labels_resNet.append(predictedLabel_resNet)

        probabilities_vector_mobileNet = mobileNet_models[i].predict(NumpyImg2Tensor(img))
        predictedLabel_mobileNet = np.argmax(probabilities_vector_mobileNet)
        predicted_labels_mobileNet.append(predictedLabel_mobileNet)

    f1_score_inception = f1_score(y_test, predicted_labels_inception, average="macro")
    f1_score_resNet = f1_score(y_test, predicted_labels_resNet, average="macro")
    f1_score_mobileNet = f1_score(y_test, predicted_labels_mobileNet, average="macro")

    f1_scores_inception.append(f1_score_inception)
    f1_scores_resNet.append(f1_score_resNet)
    f1_scores_mobileNet.append(f1_score_mobileNet)

average_f1_score_inception = np.mean(f1_scores_inception)
std_dev_f1_score_inception = np.std(f1_scores_inception)

average_f1_score_resNet = np.mean(f1_scores_resNet)
std_dev_f1_score_resNet = np.std(f1_scores_resNet)

average_f1_score_mobileNet = np.mean(f1_scores_mobileNet)
std_dev_f1_score_mobileNet = np.std(f1_scores_mobileNet)

print(f'InceptionV3: {average_f1_score_inception} ± {std_dev_f1_score_inception}')
print(f'ResNet50: {average_f1_score_resNet} ± {std_dev_f1_score_resNet}')
print(f'MobileNetV2: {average_f1_score_mobileNet} ± {std_dev_f1_score_mobileNet}')
















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
