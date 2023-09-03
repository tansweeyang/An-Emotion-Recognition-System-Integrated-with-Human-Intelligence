import os
import numpy as np
import Plotter
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score
from keras.models import load_model
from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
from HumanQLearning import HumanQLearning
from ImageHelper import NumpyImg2Tensor
from QLearningModel import QLearningModel

config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4)
session = tf.compat.v1.Session(config=config)

# ---------------------------1) Program Variables--------------------------------------
TRAIN_QL = False
num_splits = 10

train_splits = []
test_splits = []
validation_splits = []
loaded_arrays = {}
# --------------------------------------------------------------------------------------

if TRAIN_QL:
    # ---------------------------1) Load Splits------------------------------------------
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

    print('Train test split loaded.')
    # ------------------------------------------------------------------------------------

    # ---------------------------2) Load Inception Models---------------------------------
    inception_models = []
    for i in range(0, num_splits):
        inception_model = load_model(f'models/inception_model_fold_{i}.h5')
        inception_models.append(inception_model)

    inception_objects = []
    X_train = train_splits[0][0]
    image_shape = X_train[0].shape
    for i in range(0, num_splits):
        inception_object = ConvolutionalNeuralNetworks('Inception')
        inception_objects.append(inception_object)
        inception_objects[i].create_model_architecture(image_shape)
        inception_objects[i].model = inception_models[i]

    print('Inception models loaded and initialized.')
    # ------------------------------------------------------------------------------------

    # --------------------------------------3) Train QL Function For Plot------------------------------
    classes = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}

    inception_model_object = ConvolutionalNeuralNetworks('Inception')
    X_train = train_splits[0][0]
    image_shape = X_train[0].shape
    inception_model_object.create_model_architecture(image_shape)

    def train_QL(QL_network, QL_network_name, action_selection_strategy, alpha, gamma):
        max_Q_values_lists = []
        cum_r_lists = []
        predicted_labels_before_ql = []
        predicted_labels_after_ql = []

        accuracy_list_before = []
        accuracy_list_after = []
        f1_score_list_before = []
        f1_score_list_after = []

        print(f'Training {QL_network_name} using {action_selection_strategy}')
        for i in range(num_splits):
            test_split = test_splits[i]
            X_test = test_split[0]
            y_test = test_split[1]

            predicted_labels_before_ql_one_split = []
            predicted_labels_after_ql_one_split = []

            for img, label in zip(X_test, y_test):
                print(f'Split: {i}')
                # Prediction before applying action
                probabilities_vector_before_applying_action = inception_models[i].predict(NumpyImg2Tensor(img))
                predictedLabel_before_applying_action = np.argmax(probabilities_vector_before_applying_action)
                predicted_labels_before_ql_one_split.append(predictedLabel_before_applying_action)
                print(f'predicted_labels_before_ql_one_split: {predicted_labels_before_ql_one_split}')

                if (predictedLabel_before_applying_action != label):
                    QL_network.perform_iterative_Q_learning(inception_model_object, img, classes, action_selection_strategy, alpha, gamma)
                    optimal_action = QL_network.choose_optimal_action()
                    corrected_img = QL_network.apply_action(optimal_action, img)

                    probabilities_vector_after_applying_action = inception_models[i].predict(NumpyImg2Tensor(corrected_img))
                    # Prediction after applying action
                    predictedLabel_after_applying_action = np.argmax(probabilities_vector_after_applying_action)
                    predicted_labels_after_ql_one_split.append(predictedLabel_after_applying_action)
                    print(f'predicted_labels_after_ql_one_split: {predicted_labels_after_ql_one_split}')
                else:
                    predicted_labels_after_ql_one_split.append(predictedLabel_before_applying_action)
                    print(f'predictedLabel_before_applying_action: {predicted_labels_after_ql_one_split}')

            accuracy_one_fold_before_ql = accuracy_score(y_test, predicted_labels_before_ql_one_split)
            f1_score_one_fold_before_ql = f1_score(y_test, predicted_labels_before_ql_one_split, average="macro")
            accuracy_one_fold_after_ql = accuracy_score(y_test, predicted_labels_after_ql_one_split)
            f1_score_one_fold_after_ql = f1_score(y_test, predicted_labels_after_ql_one_split, average="macro")

            accuracy_list_before.append(accuracy_one_fold_before_ql)
            f1_score_list_before.append(f1_score_one_fold_before_ql)
            accuracy_list_after.append(accuracy_one_fold_after_ql)
            f1_score_list_after.append(f1_score_one_fold_after_ql)

            predicted_labels_before_ql.append(predicted_labels_before_ql_one_split)
            predicted_labels_after_ql.append(predicted_labels_after_ql_one_split)

            # Output of one split [1,2,3,4,5]
            max_Q_values_TS_QL_list = QL_network.get_best_max_Q_values_one_img()
            print(f'max_Q_values_TS_QL_list: {max_Q_values_TS_QL_list}')

            cum_r_TS_QL_list = QL_network.get_best_max_cum_r_one_img()
            cum_r_TS_QL_list = cum_r_TS_QL_list.tolist()
            print(f'cum_r_TS_QL_list: {cum_r_TS_QL_list}')

            max_Q_values_lists.append(max_Q_values_TS_QL_list)
            print(f'max_Q_values_lists appended: {max_Q_values_lists}')

            cum_r_lists.append(cum_r_TS_QL_list)
            print(f'cum_r_lists appended: {cum_r_lists}')

        average_max_Q_values_list = np.mean(max_Q_values_lists, axis=0)
        print(f'Average max Q values {QL_network_name} list: {average_max_Q_values_list}')
        std_dev_max_Q_values_list = np.std(max_Q_values_lists, axis=0)
        print(f'Std dev max Q values {QL_network_name} list: {std_dev_max_Q_values_list}')

        average_cum_r_list = np.mean(cum_r_lists, axis=0)
        print(f'Average cumulative reward {QL_network_name} list: {average_cum_r_list}')
        std_dev_cum_r_list = np.std(cum_r_lists, axis=0)
        print(f'Std dev cumulative reward {QL_network_name} list: {std_dev_cum_r_list}')

        # get accuracy and return it
        print(f'predicted_labels_before_ql: {predicted_labels_before_ql}')
        print(f'predicted_labels_after_ql: {predicted_labels_after_ql}')

        print(f'accuracy_list_before: {accuracy_list_before}')
        print(f'f1_score_list_before: {f1_score_list_before}')
        print(f'accuracy_list_after: {accuracy_list_after}')
        print(f'f1_score_list_after: {f1_score_list_after}')

        final_accuracy_before = f'{np.mean(accuracy_list_before)} ± {np.std(accuracy_list_before)}'
        final_f1_score_before = f'{np.mean(f1_score_list_before)} ± {np.std(f1_score_list_before)}'
        final_accuracy_after = f'{np.mean(accuracy_list_after)}  ± {np.std(accuracy_list_after)}'
        final_f1_score_after = f'{np.mean(f1_score_list_after)} ± {np.std(f1_score_list_after)}'

        return average_max_Q_values_list, std_dev_max_Q_values_list, average_cum_r_list, std_dev_cum_r_list, final_accuracy_before, final_f1_score_before, final_accuracy_after, final_f1_score_after
    # -----------------------------------------------------------------------------------------------------

    # -------------------------------------4) Variables for Entire Plot--------------------------------------------------
    random_strategy = 'random'
    harmonic_decay_strategy = 'harmonic-sequence-e-decay'
    one_shot_decay_strategy = 'one-shot-e-decay'

    TS_QL_random = QLearningModel()
    TS_QL_harmonic = QLearningModel()
    TS_QL_one_shot = QLearningModel()
    # ------------------------------------------------------------------------------------------------------------------

    # -----------------------------------5) Train QL for Subplot (0,0) (0,1)--------------------------------------------
    average_max_Q_values_TS_QL_list_random, std_dev_max_Q_values_TS_QL_list_random, average_cum_r_TS_QL_list_random, std_dev_cum_r_TS_QL_list_random, final_accuracy_before_TS_QL_random, final_f1_score_before_TS_QL_random, final_accuracy_after_TS_QL_random, final_f1_score_after_TS_QL_random = train_QL(TS_QL_random, 'TS-QL', random_strategy, 1.0, 0.0)
    average_max_Q_values_TS_QL_list_harmonic, std_dev_max_Q_values_TS_QL_list_harmonic, average_cum_r_TS_QL_list_harmonic, std_dev_cum_r_TS_QL_list_harmonic, final_accuracy_before_TS_QL_harmonic, final_f1_score_before_TS_QL_harmonic, final_accuracy_after_TS_QL_harmonic, final_f1_score_after_TS_QL_harmonic = train_QL(TS_QL_harmonic, 'TS-QL', harmonic_decay_strategy, 1.0, 0.0)
    average_max_Q_values_TS_QL_list_one_shot, std_dev_max_Q_values_TS_QL_list_one_shot, average_cum_r_TS_QL_list_one_shot, std_dev_cum_r_TS_QL_list_one_shot, final_accuracy_before_TS_QL_one_shot, final_f1_score_before_TS_QL_one_shot, final_accuracy_after_TS_QL_one_shot, final_f1_score_after_TS_QL_one_shot = train_QL(TS_QL_one_shot, 'TS-QL', one_shot_decay_strategy, 1.0, 0.0)

    # Subplot (0,0)
    print(f'\naverage_max_Q_values_TS_QL_list_random: {average_max_Q_values_TS_QL_list_random}')
    print(f'std_dev_max_Q_values_TS_QL_list_random: {std_dev_max_Q_values_TS_QL_list_random}')
    print(f'average_max_Q_values_TS_QL_list_harmonic: {average_max_Q_values_TS_QL_list_harmonic}')
    print(f'std_dev_max_Q_values_TS_QL_list_harmonic: {std_dev_max_Q_values_TS_QL_list_harmonic}')
    print(f'average_max_Q_values_TS_QL_list_one_shot: {average_max_Q_values_TS_QL_list_one_shot}')
    print(f'std_dev_max_Q_values_TS_QL_list_one_shot: {std_dev_max_Q_values_TS_QL_list_one_shot}\n')

    # Subplot (0,1)
    print(f'average_cum_r_TS_QL_list_random: {average_cum_r_TS_QL_list_random}')
    print(f'std_dev_cum_r_TS_QL_list_random: {std_dev_cum_r_TS_QL_list_random}')
    print(f'average_cum_r_TS_QL_list_harmonic: {average_cum_r_TS_QL_list_harmonic}')
    print(f'std_dev_cum_r_TS_QL_list_harmonic: {std_dev_cum_r_TS_QL_list_harmonic}')
    print(f'average_cum_r_TS_QL_list_one_shot: {average_cum_r_TS_QL_list_one_shot}')
    print(f'std_dev_cum_r_TS_QL_list_one_shot: {std_dev_cum_r_TS_QL_list_one_shot}\n')

    # Save these
    print(f'final_accuracy_before_TS_QL_random: {final_accuracy_before_TS_QL_random}')
    print(f'final_f1_score_before_TS_QL_random: {final_f1_score_before_TS_QL_random}')
    print(f'final_accuracy_after_TS_QL_random: {final_accuracy_after_TS_QL_random}')
    print(f'final_f1_score_after_TS_QL_random: {final_f1_score_after_TS_QL_random}')
    print(f'final_accuracy_before_TS_QL_harmonic: {final_accuracy_before_TS_QL_harmonic}')
    print(f'final_f1_score_before_TS_QL_harmonic: {final_f1_score_before_TS_QL_harmonic}')
    print(f'final_accuracy_after_TS_QL_harmonic: {final_accuracy_after_TS_QL_harmonic}')
    print(f'final_f1_score_after_TS_QL_harmonic: {final_f1_score_after_TS_QL_harmonic}')
    print(f'final_accuracy_before_TS_QL_one_shot: {final_accuracy_before_TS_QL_one_shot}')
    print(f'final_f1_score_before_TS_QL_one_shot: {final_f1_score_before_TS_QL_one_shot}')
    print(f'final_accuracy_after_TS_QL_one_shot: {final_accuracy_after_TS_QL_one_shot}')
    print(f'final_f1_score_after_TS_QL_one_shot: {final_f1_score_after_TS_QL_one_shot}\n')

    final_score_for_TS_QL_random = [final_accuracy_before_TS_QL_random, final_f1_score_before_TS_QL_random,
                                    final_accuracy_after_TS_QL_random, final_f1_score_after_TS_QL_random]
    final_score_for_TS_QL_harmonic = [final_accuracy_before_TS_QL_harmonic, final_f1_score_before_TS_QL_harmonic,
                                      final_accuracy_after_TS_QL_harmonic, final_f1_score_after_TS_QL_harmonic]
    final_score_for_TS_QL_one_shot = [final_accuracy_before_TS_QL_one_shot, final_f1_score_before_TS_QL_one_shot,
                                      final_accuracy_after_TS_QL_one_shot, final_f1_score_after_TS_QL_one_shot]

    print('Training QL for subplot (0,0) (0,1) completed.\n')
    # ------------------------------------------------------------------------------------------------------

    # -----------------------------------6) Train QL for Subplot (1,0) (1,1)---------------------------------
    TS_QL_HF_random = HumanQLearning()
    TS_QL_HF_harmonic = HumanQLearning()
    TS_QL_HF_one_shot = HumanQLearning()

    average_max_Q_values_TS_QL_HF_list_random, std_dev_max_Q_values_TS_QL_HF_list_random, average_cum_r_TS_QL_HF_list_random, std_dev_cum_r_TS_QL_HF_list_random, final_accuracy_before_TS_QL_HF_random, final_f1_score_before_TS_QL_HF_random, final_accuracy_after_TS_QL_HF_random, final_f1_score_after_TS_QL_HF_random = train_QL(TS_QL_HF_random, 'TS-QL-HF', random_strategy, 1.0, 0.0)
    average_max_Q_values_TS_QL_HF_list_harmonic, std_dev_max_Q_values_TS_QL_HF_list_harmonic, average_cum_r_TS_QL_HF_list_harmonic, std_dev_cum_r_TS_QL_HF_list_harmonic, final_accuracy_before_TS_QL_HF_harmonic, final_f1_score_before_TS_QL_HF_harmonic, final_accuracy_after_TS_QL_HF_harmonic, final_f1_score_after_TS_QL_HF_harmonic = train_QL(TS_QL_HF_harmonic, 'TS-QL-HF', harmonic_decay_strategy, 1.0, 0.0)
    average_max_Q_values_TS_QL_HF_list_one_shot, std_dev_max_Q_values_TS_QL_HF_list_one_shot, average_cum_r_TS_QL_HF_list_one_shot, std_dev_cum_r_TS_QL_HF_list_one_shot, final_accuracy_before_TS_QL_HF_one_shot, final_f1_score_before_TS_QL_HF_one_shot, final_accuracy_after_TS_QL_HF_one_shot, final_f1_score_after_TS_QL_HF_one_shot = train_QL(TS_QL_HF_one_shot, 'TS-QL-HF', one_shot_decay_strategy, 1.0, 0.0)

    # Subplot (1,0)
    print(f'\naverage_max_Q_values_TS_QL_HF_list_random: {average_max_Q_values_TS_QL_HF_list_random}')
    print(f'std_dev_max_Q_values_TS_QL_HF_list_random: {std_dev_max_Q_values_TS_QL_HF_list_random}')
    print(f'average_max_Q_values_TS_QL_HF_list_harmonic: {average_max_Q_values_TS_QL_HF_list_harmonic}')
    print(f'std_dev_max_Q_values_TS_QL_HF_list_harmonic: {std_dev_max_Q_values_TS_QL_HF_list_harmonic}')
    print(f'average_max_Q_values_TS_QL_HF_list_one_shot: {average_max_Q_values_TS_QL_HF_list_one_shot}')
    print(f'std_dev_max_Q_values_TS_QL_HF_list_one_shot: {std_dev_max_Q_values_TS_QL_HF_list_one_shot}\n')

    # Subplot (1,1)
    print(f'average_cum_r_TS_QL_HF_list_random: {average_cum_r_TS_QL_HF_list_random}')
    print(f'std_dev_cum_r_TS_QL_HF_list_random: {std_dev_cum_r_TS_QL_HF_list_random}')
    print(f'average_cum_r_TS_QL_HF_list_harmonic: {average_cum_r_TS_QL_HF_list_harmonic}')
    print(f'std_dev_cum_r_TS_QL_HF_list_harmonic: {std_dev_cum_r_TS_QL_HF_list_harmonic}')
    print(f'average_cum_r_TS_QL_HF_list_one_shot: {average_cum_r_TS_QL_HF_list_one_shot}')
    print(f'std_dev_cum_r_TS_QL_HF_list_one_shot: {std_dev_cum_r_TS_QL_HF_list_one_shot}\n')

    # Save these
    print(f'final_accuracy_before_TS_QL_HF_random: {final_accuracy_before_TS_QL_HF_random}')
    print(f'final_f1_score_before_TS_QL_HF_random: {final_f1_score_before_TS_QL_HF_random}')
    print(f'final_accuracy_after_TS_QL_HF_random: {final_accuracy_after_TS_QL_HF_random}')
    print(f'final_f1_score_after_TS_QL_HF_random: {final_f1_score_after_TS_QL_HF_random}')
    print(f'final_accuracy_before_TS_QL_HF_harmonic: {final_accuracy_before_TS_QL_HF_harmonic}')
    print(f'final_f1_score_before_TS_QL_HF_harmonic: {final_f1_score_before_TS_QL_HF_harmonic}')
    print(f'final_accuracy_after_TS_QL_HF_harmonic: {final_accuracy_after_TS_QL_HF_harmonic}')
    print(f'final_f1_score_after_TS_QL_HF_harmonic: {final_f1_score_after_TS_QL_HF_harmonic}')
    print(f'final_accuracy_before_TS_QL_HF_one_shot: {final_accuracy_before_TS_QL_HF_one_shot}')
    print(f'final_f1_score_before_TS_QL_HF_one_shot: {final_f1_score_before_TS_QL_HF_one_shot}')
    print(f'final_accuracy_after_TS_QL_HF_one_shot: {final_accuracy_after_TS_QL_HF_one_shot}')
    print(f'final_f1_score_after_TS_QL_HF_one_shot: {final_f1_score_after_TS_QL_HF_one_shot}\n')

    final_score_for_TS_QL_HF_random = [final_accuracy_before_TS_QL_HF_random, final_f1_score_before_TS_QL_HF_random,
                                       final_accuracy_after_TS_QL_HF_random, final_f1_score_after_TS_QL_HF_random]
    final_score_for_TS_QL_HF_harmonic = [final_accuracy_before_TS_QL_HF_harmonic, final_f1_score_before_TS_QL_HF_harmonic,
                                         final_accuracy_after_TS_QL_HF_harmonic, final_f1_score_after_TS_QL_HF_harmonic]
    final_score_for_TS_QL_HF_one_shot = [final_accuracy_before_TS_QL_HF_one_shot, final_f1_score_before_TS_QL_HF_one_shot,
                                         final_accuracy_after_TS_QL_HF_one_shot, final_f1_score_after_TS_QL_HF_one_shot]

    print('Training QL for subplot (1,0) (1,1) completed.\n')
    # ------------------------------------------------------------------------------------------------------

    # ---------------------------7) Save output arrays------------------------------------
    # Define the directory where you want to save the data
    save_directory = "QL_Plot1_Train_Data"

    # Define the array names and corresponding data
    array_data = {
        "average_max_Q_values_TS_QL_list_random": average_max_Q_values_TS_QL_list_random,
        "std_dev_max_Q_values_TS_QL_list_random": std_dev_max_Q_values_TS_QL_list_random,
        "average_max_Q_values_TS_QL_list_harmonic": average_max_Q_values_TS_QL_list_harmonic,
        "std_dev_max_Q_values_TS_QL_list_harmonic": std_dev_max_Q_values_TS_QL_list_harmonic,
        "average_max_Q_values_TS_QL_list_one_shot": average_max_Q_values_TS_QL_list_one_shot,
        "std_dev_max_Q_values_TS_QL_list_one_shot": std_dev_max_Q_values_TS_QL_list_one_shot,

        "average_cum_r_TS_QL_list_random": average_cum_r_TS_QL_list_random,
        "std_dev_cum_r_TS_QL_list_random": std_dev_cum_r_TS_QL_list_random,
        "average_cum_r_TS_QL_list_harmonic": average_cum_r_TS_QL_list_harmonic,
        "std_dev_cum_r_TS_QL_list_harmonic": std_dev_cum_r_TS_QL_list_harmonic,
        "average_cum_r_TS_QL_list_one_shot": average_cum_r_TS_QL_list_one_shot,
        "std_dev_cum_r_TS_QL_list_one_shot": std_dev_cum_r_TS_QL_list_one_shot,

        "average_max_Q_values_TS_QL_HF_list_random": average_max_Q_values_TS_QL_HF_list_random,
        "std_dev_max_Q_values_TS_QL_HF_list_random": std_dev_max_Q_values_TS_QL_HF_list_random,
        "average_max_Q_values_TS_QL_HF_list_harmonic": average_max_Q_values_TS_QL_HF_list_harmonic,
        "std_dev_max_Q_values_TS_QL_HF_list_harmonic": std_dev_max_Q_values_TS_QL_HF_list_harmonic,
        "average_max_Q_values_TS_QL_HF_list_one_shot": average_max_Q_values_TS_QL_HF_list_one_shot,
        "std_dev_max_Q_values_TS_QL_HF_list_one_shot": std_dev_max_Q_values_TS_QL_HF_list_one_shot,

        "average_cum_r_TS_QL_HF_list_random": average_cum_r_TS_QL_HF_list_random,
        "std_dev_cum_r_TS_QL_HF_list_random": std_dev_cum_r_TS_QL_HF_list_random,
        "average_cum_r_TS_QL_HF_list_harmonic": average_cum_r_TS_QL_HF_list_harmonic,
        "std_dev_cum_r_TS_QL_HF_list_harmonic": std_dev_cum_r_TS_QL_HF_list_harmonic,
        "average_cum_r_TS_QL_HF_list_one_shot": average_cum_r_TS_QL_HF_list_one_shot,
        "std_dev_cum_r_TS_QL_HF_list_one_shot": std_dev_cum_r_TS_QL_HF_list_one_shot,

        "final_score_for_TS_QL_random": final_score_for_TS_QL_random,
        "final_score_for_TS_QL_harmonic": final_score_for_TS_QL_harmonic,
        "final_score_for_TS_QL_one_shot": final_score_for_TS_QL_one_shot,

        "final_score_for_TS_QL_HF_random": final_score_for_TS_QL_HF_random,
        "final_score_for_TS_QL_HF_harmonic": final_score_for_TS_QL_HF_harmonic,
        "final_score_for_TS_QL_HF_one_shot": final_score_for_TS_QL_HF_one_shot
    }

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save each array using np.save and organize them in the directory structure
    for array_name, array in array_data.items():
        file_path = os.path.join(save_directory, f"{array_name}.npy")
        np.save(file_path, array)
        print(f"Saved {array_name} to {file_path}")
    # ------------------------------------------------------------------------------------

# ---------------------------8) Load output arrays------------------------------------
array_names = {
    "average_max_Q_values_TS_QL_list_random", "std_dev_max_Q_values_TS_QL_list_random",
    "average_max_Q_values_TS_QL_list_harmonic", "std_dev_max_Q_values_TS_QL_list_harmonic",
    "average_max_Q_values_TS_QL_list_one_shot", "std_dev_max_Q_values_TS_QL_list_one_shot",
    "average_cum_r_TS_QL_list_random", "std_dev_cum_r_TS_QL_list_random",
    "average_cum_r_TS_QL_list_harmonic", "std_dev_cum_r_TS_QL_list_harmonic",
    "average_cum_r_TS_QL_list_one_shot", "std_dev_cum_r_TS_QL_list_one_shot",
    "average_max_Q_values_TS_QL_HF_list_random", "std_dev_max_Q_values_TS_QL_HF_list_random",
    "average_max_Q_values_TS_QL_HF_list_harmonic", "std_dev_max_Q_values_TS_QL_HF_list_harmonic",
    "average_max_Q_values_TS_QL_HF_list_one_shot", "std_dev_max_Q_values_TS_QL_HF_list_one_shot",
    "average_cum_r_TS_QL_HF_list_random", "std_dev_cum_r_TS_QL_HF_list_random",
    "average_cum_r_TS_QL_HF_list_harmonic", "std_dev_cum_r_TS_QL_HF_list_harmonic",
    "average_cum_r_TS_QL_HF_list_one_shot", "std_dev_cum_r_TS_QL_HF_list_one_shot",
                                          
    "final_score_for_TS_QL_random", "final_score_for_TS_QL_harmonic", "final_score_for_TS_QL_one_shot",
    "final_score_for_TS_QL_HF_random", "final_score_for_TS_QL_HF_harmonic", "final_score_for_TS_QL_HF_one_shot"
}

load_directory = "QL_Plot1_Train_Data"

for array_name in array_names:
    file_path = os.path.join(load_directory, f"{array_name}.npy")
    loaded_array = np.load(file_path, allow_pickle=True)
    loaded_arrays[array_name] = loaded_array
    print(f"Loaded {array_name} from {file_path}")
# --------------------------------------------------------------------------------

# -------------------------------------------------9) Plot the graph ------------------------------------------------------------------------------------
episodes = range(1, 11)
Plotter.plotActionSelectionAnalysis(episodes,
                            loaded_arrays["average_max_Q_values_TS_QL_list_random"], loaded_arrays["std_dev_max_Q_values_TS_QL_list_random"],
                            loaded_arrays["average_max_Q_values_TS_QL_list_harmonic"], loaded_arrays["std_dev_max_Q_values_TS_QL_list_harmonic"],
                            loaded_arrays["average_max_Q_values_TS_QL_list_one_shot"], loaded_arrays["std_dev_max_Q_values_TS_QL_list_one_shot"],
                            loaded_arrays["average_cum_r_TS_QL_list_random"], loaded_arrays["std_dev_cum_r_TS_QL_list_random"],
                            loaded_arrays["average_cum_r_TS_QL_list_harmonic"], loaded_arrays["std_dev_cum_r_TS_QL_list_harmonic"],
                            loaded_arrays["average_cum_r_TS_QL_list_one_shot"], loaded_arrays["std_dev_cum_r_TS_QL_list_one_shot"],
                            loaded_arrays["average_max_Q_values_TS_QL_HF_list_random"], loaded_arrays["std_dev_max_Q_values_TS_QL_HF_list_random"],
                            loaded_arrays["average_max_Q_values_TS_QL_HF_list_harmonic"], loaded_arrays["std_dev_max_Q_values_TS_QL_HF_list_harmonic"],
                            loaded_arrays["average_max_Q_values_TS_QL_HF_list_one_shot"], loaded_arrays["std_dev_max_Q_values_TS_QL_HF_list_one_shot"],
                            loaded_arrays["average_cum_r_TS_QL_HF_list_random"], loaded_arrays["std_dev_cum_r_TS_QL_HF_list_random"],
                            loaded_arrays["average_cum_r_TS_QL_HF_list_harmonic"], loaded_arrays["std_dev_cum_r_TS_QL_HF_list_harmonic"],
                            loaded_arrays["average_cum_r_TS_QL_HF_list_one_shot"], loaded_arrays["std_dev_cum_r_TS_QL_HF_list_one_shot"])
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------- Print accuracy before and after -------------------#
final_score_for_TS_QL_random = loaded_arrays["final_score_for_TS_QL_random"]
final_score_for_TS_QL_harmonic = loaded_arrays["final_score_for_TS_QL_harmonic"]
final_score_for_TS_QL_one_shot = loaded_arrays["final_score_for_TS_QL_one_shot"]

final_score_for_TS_QL_HF_random = loaded_arrays["final_score_for_TS_QL_HF_random"]
final_score_for_TS_QL_HF_harmonic = loaded_arrays["final_score_for_TS_QL_HF_harmonic"]
final_score_for_TS_QL_HF_one_shot = loaded_arrays["final_score_for_TS_QL_HF_one_shot"]

print(f'Accuracy after TS_QL_one_shot: {final_score_for_TS_QL_one_shot[2]}')
print(f'Final f1_score after TS_QL_one_shot: {final_score_for_TS_QL_one_shot[3]}\n')

print(f'Accuracy after TS_QL_harmonic: {final_score_for_TS_QL_harmonic[2]}')
print(f'Final f1_score after TS_QL_harmonic: {final_score_for_TS_QL_harmonic[3]}\n')

print(f'Accuracy after TS_QL_random: {final_score_for_TS_QL_random[2]}')
print(f'Final f1_score after TS_QL_random: {final_score_for_TS_QL_random[3]}\n')

print(f'Accuracy after TS_QL_HF_one_shot: {final_score_for_TS_QL_HF_one_shot[2]}')
print(f'Final f1_score after TS_QL_HF_one_shot: {final_score_for_TS_QL_HF_one_shot[3]}\n')

print(f'Accuracy after TS_QL_HF_harmonic: {final_score_for_TS_QL_HF_harmonic[2]}')
print(f'Final f1_score after TS_QL_HF_harmonic: {final_score_for_TS_QL_HF_harmonic[3]}\n')

print(f'Accuracy after TS_QL_HF_random: {final_score_for_TS_QL_HF_random[2]}')
print(f'Final f1_score after TS_QL_HF_random: {final_score_for_TS_QL_HF_random[3]}\n')