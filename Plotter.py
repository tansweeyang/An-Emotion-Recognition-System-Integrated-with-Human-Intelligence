import os
import seaborn as sn
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

def plot_model_history(average_accuracy_list_inception_model, average_val_accuracy_list_inception_model,
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
                        std_deviation_loss_list_mobileNet_model, std_deviation_val_loss_list_mobileNet_model):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
    epochs = range(1, len(average_accuracy_list_inception_model) + 1)
    y_ticks_accuracy = np.arange(0, 1.1, 0.2)
    y_ticks_loss = np.arange(-400, 900, 200)

    axes[0, 0].set_title('(a) InceptionV3')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].plot(epochs, average_accuracy_list_inception_model, color='red', label='train')
    axes[0, 0].fill_between(epochs, average_accuracy_list_inception_model - std_deviation_accuracy_list_inception_model,
                            average_accuracy_list_inception_model + std_deviation_accuracy_list_inception_model,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[0, 0].plot(epochs, average_val_accuracy_list_inception_model, color='orange', label='validation')
    axes[0, 0].fill_between(epochs, average_val_accuracy_list_inception_model - std_deviation_val_accuracy_list_inception_model,
                            average_val_accuracy_list_inception_model + std_deviation_val_accuracy_list_inception_model,
                            color='orange', alpha=0.2, label='_nolegend_')
    axes[0, 0].set_yticks(y_ticks_accuracy)

    axes[0, 1].set_title('(b) ResNet50')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].plot(epochs, average_accuracy_list_resNet_model, color='red', label='train')
    axes[0, 1].fill_between(epochs, average_accuracy_list_resNet_model - std_deviation_accuracy_list_resNet_model,
                            average_accuracy_list_resNet_model + std_deviation_accuracy_list_resNet_model,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[0, 1].plot(epochs, average_val_accuracy_list_resNet_model, color='orange', label='validation')
    axes[0, 1].fill_between(epochs, average_val_accuracy_list_resNet_model - std_deviation_val_accuracy_list_resNet_model,
                            average_val_accuracy_list_resNet_model + std_deviation_val_accuracy_list_resNet_model,
                            color='orange', alpha=0.2, label='_nolegend_')
    axes[0, 1].set_yticks(y_ticks_accuracy)

    axes[0, 2].set_title('(c) MobileNetV2')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].plot(epochs, average_accuracy_list_mobileNet_model, color='red', label='train')
    axes[0, 2].fill_between(epochs, average_accuracy_list_mobileNet_model - std_deviation_accuracy_list_mobileNet_model,
                            average_accuracy_list_mobileNet_model + std_deviation_accuracy_list_resNet_model,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[0, 2].plot(epochs, average_val_accuracy_list_mobileNet_model, color='orange', label='validation')
    axes[0, 2].fill_between(epochs, average_val_accuracy_list_mobileNet_model - std_deviation_val_accuracy_list_mobileNet_model,
                           average_val_accuracy_list_mobileNet_model + std_deviation_val_accuracy_list_mobileNet_model,
                           color='orange', alpha=0.2, label='_nolegend_')
    axes[0, 2].set_yticks(y_ticks_accuracy)

    axes[1, 0].set_title('(d) InceptionV3')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].plot(epochs, average_loss_list_inception_model, color='red', label='train')
    axes[1, 0].fill_between(epochs, average_loss_list_inception_model - std_deviation_loss_list_inception_model,
                            average_loss_list_inception_model + std_deviation_loss_list_inception_model,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[1, 0].plot(epochs, average_val_loss_list_inception_model, color='orange', label='validation')
    axes[1, 0].fill_between(epochs, average_val_loss_list_inception_model - std_deviation_val_loss_list_inception_model,
                            average_val_loss_list_inception_model + std_deviation_val_loss_list_inception_model,
                            color='orange', alpha=0.2, label='_nolegend_')
    axes[1, 0].set_yticks(y_ticks_loss)

    axes[1, 1].set_title('(e) ResNet50')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].plot(epochs, average_loss_list_resNet_model, color='red', label='train')
    axes[1, 1].fill_between(epochs, average_loss_list_resNet_model - std_deviation_loss_list_resNet_model,
                            average_loss_list_resNet_model + std_deviation_loss_list_resNet_model,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[1, 1].plot(epochs, average_val_loss_list_resNet_model, color='orange', label='validation')
    axes[1, 1].fill_between(epochs, average_val_loss_list_resNet_model - std_deviation_val_loss_list_resNet_model,
                            average_val_loss_list_resNet_model + std_deviation_val_loss_list_resNet_model,
                            color='orange', alpha=0.2, label='_nolegend_')
    axes[1, 1].set_yticks(y_ticks_loss)

    axes[1, 2].set_title('(f) MobileNetV2')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].plot(epochs, average_loss_list_mobileNet_model, color='red', label='train')
    axes[1, 2].fill_between(epochs, average_loss_list_mobileNet_model - std_deviation_loss_list_mobileNet_model,
                            average_loss_list_mobileNet_model + std_deviation_loss_list_mobileNet_model,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[1, 2].plot(epochs, average_val_loss_list_mobileNet_model, color='orange', label='validation')
    axes[1, 2].fill_between(epochs, average_val_loss_list_mobileNet_model - std_deviation_val_loss_list_mobileNet_model,
                            average_val_loss_list_mobileNet_model + std_deviation_val_loss_list_mobileNet_model,
                            color='orange', alpha=0.2, label='_nolegend_')
    axes[1, 2].set_yticks(y_ticks_loss)

    plt.figlegend(['train', 'validation'], loc='lower center')
    plt.tight_layout()
    plt.show()

# TS-QL Fold 1 Cum_reward : [1, 2, 3, 4, 5]
# TS-QL Fold 2 Cum_reward : [1, 2, 3, 4, 5]
# TS-QL Fold 3 Cum_reward : [1, 2, 3, 4, 5]
# TS-QL Fold 4 Cum_reward : [1, 2, 3, 4, 5]
# To pass in: TS-QL Cum_reward (Averaged) : [1, 2, 3, 4, 5]

def plot_QL_history(episodes,
                    average_max_Q_values_TS_QL_list, std_dev_max_Q_values_TS_QL_list, average_cum_r_TS_QL_list, std_dev_cum_r_TS_QL_list,
                    average_max_Q_values_TS_QL_HF_list, std_dev_max_Q_values_TS_QL_HF_list, average_cum_r_TS_QL_HF_list, std_dev_cum_r_TS_QL_HF_list,
                    average_max_Q_values_TS_DQL_HF_list, std_dev_max_Q_values_TS_DQL_HF_list, average_cum_r_TS_DQL_HF_list, std_dev_cum_r_TS_DQL_HF_list):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    axes[0].set_title('(a) Max Q-values of First Misclassified Image Averaged Over 10 Folds')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Max Q-values')
    axes[0].plot(episodes, average_max_Q_values_TS_QL_list, color='blue', label='TS-QL')
    axes[0].fill_between(episodes, average_max_Q_values_TS_QL_list - std_dev_max_Q_values_TS_QL_list,
                            average_max_Q_values_TS_QL_list + std_dev_max_Q_values_TS_QL_list,
                            color='blue', alpha=0.2, label='_nolegend_')
    axes[0].plot(episodes, average_max_Q_values_TS_QL_HF_list, color='orange', label='TS-QL-HF')
    axes[0].fill_between(episodes, average_max_Q_values_TS_QL_HF_list - std_dev_max_Q_values_TS_QL_HF_list,
                         average_max_Q_values_TS_QL_HF_list + std_dev_max_Q_values_TS_QL_HF_list,
                         color='orange', alpha=0.2, label='_nolegend_')
    axes[0].plot(episodes, average_max_Q_values_TS_DQL_HF_list, color='red', label='TS-DQL-HF')
    axes[0].fill_between(episodes, average_max_Q_values_TS_DQL_HF_list - std_dev_max_Q_values_TS_DQL_HF_list,
                         average_max_Q_values_TS_DQL_HF_list + std_dev_max_Q_values_TS_DQL_HF_list,
                         color='red', alpha=0.2, label='_nolegend_')

    axes[1].set_title('(b) Cumulative Reward of First Misclassified Image Averaged Over 10 Folds')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Cumulative Reward')
    axes[1].plot(episodes, average_cum_r_TS_QL_list, color='blue', label='TS-QL')
    axes[1].fill_between(episodes, average_cum_r_TS_QL_list - std_dev_cum_r_TS_QL_list,
                         average_cum_r_TS_QL_list + std_dev_cum_r_TS_QL_list,
                         color='blue', alpha=0.2, label='_nolegend_')
    axes[1].plot(episodes, average_cum_r_TS_QL_HF_list, color='orange', label='TS-QL-HF')
    axes[1].fill_between(episodes, average_cum_r_TS_QL_HF_list - std_dev_cum_r_TS_QL_HF_list,
                         average_cum_r_TS_QL_HF_list + std_dev_cum_r_TS_QL_HF_list,
                         color='blue', alpha=0.2, label='_nolegend_')
    axes[1].plot(episodes, average_cum_r_TS_DQL_HF_list, color='red', label='TS-DQL-HF')
    axes[1].fill_between(episodes, average_cum_r_TS_DQL_HF_list - std_dev_cum_r_TS_DQL_HF_list,
                         average_cum_r_TS_DQL_HF_list + std_dev_cum_r_TS_DQL_HF_list,
                         color='red', alpha=0.2, label='_nolegend_')

    plt.figlegend(['TS-QL', 'TS-QL-HF', 'TS-DQL-HF'], loc='lower center')
    plt.tight_layout()
    plt.show()

def plotActionSelectionAnalysis(episodes,
                                # (0,0)
                                average_max_Q_values_TS_QL_list_random, std_dev_max_Q_values_TS_QL_list_random,
                                average_max_Q_values_TS_QL_list_harmonic, std_dev_max_Q_values_TS_QL_list_harmonic,
                                average_max_Q_values_TS_QL_list_one_shot, std_dev_max_Q_values_TS_QL_list_one_shot,

                                # (0,1)
                                average_cum_r_TS_QL_list_random, std_dev_cum_r_TS_QL_list_random,
                                average_cum_r_TS_QL_list_harmonic, std_dev_cum_r_TS_QL_list_harmonic,
                                average_cum_r_TS_QL_list_one_shot, std_dev_cum_r_TS_QL_list_one_shot,

                                # (1,0)
                                average_max_Q_values_TS_QL_HF_list_random, std_dev_max_Q_values_TS_QL_HF_list_random,
                                average_max_Q_values_TS_QL_HF_list_harmonic, std_dev_max_Q_values_TS_QL_HF_list_harmonic,
                                average_max_Q_values_TS_QL_HF_list_one_shot, std_dev_max_Q_values_TS_QL_HF_list_one_shot,

                                # (1,1)
                                average_cum_r_TS_QL_HF_list_random, std_dev_cum_r_TS_QL_HF_list_random,
                                average_cum_r_TS_QL_HF_list_harmonic, std_dev_cum_r_TS_QL_HF_list_harmonic,
                                average_cum_r_TS_QL_HF_list_one_shot, std_dev_cum_r_TS_QL_HF_list_one_shot
                                ):

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    y_ticks_cum_r = np.arange(-40, 60, 20)

    axes[0, 0].set_title('(a) Max Q-Values of Best Misclassified Image in TS-QL Averaged over 10 Folds', fontsize=8)
    axes[0, 0].set_xlabel('Episodes', fontsize=8)
    axes[0, 0].set_ylabel('Max Q-values', fontsize=8)
    axes[0, 0].plot(episodes, average_max_Q_values_TS_QL_list_one_shot, color='red', label='oneshot e-decay')
    axes[0, 0].fill_between(episodes,
                            average_max_Q_values_TS_QL_list_one_shot - std_dev_max_Q_values_TS_QL_list_one_shot,
                            average_max_Q_values_TS_QL_list_one_shot + std_dev_max_Q_values_TS_QL_list_one_shot,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[0, 0].plot(episodes, average_max_Q_values_TS_QL_list_harmonic, color='orange', label='harmonic e-decay')
    axes[0, 0].fill_between(episodes,
                            average_max_Q_values_TS_QL_list_harmonic - std_dev_max_Q_values_TS_QL_list_harmonic,
                            average_max_Q_values_TS_QL_list_harmonic + std_dev_max_Q_values_TS_QL_list_harmonic,
                            color='orange', alpha=0.2, label='_nolegend_')
    axes[0, 0].plot(episodes, average_max_Q_values_TS_QL_list_random, color='blue', label='random selection')
    axes[0, 0].fill_between(episodes, average_max_Q_values_TS_QL_list_random - std_dev_max_Q_values_TS_QL_list_random,
                            average_max_Q_values_TS_QL_list_random + std_dev_max_Q_values_TS_QL_list_random,
                            color='blue', alpha=0.2, label='_nolegend_')

    axes[0, 1].set_title('(b) Cumulative Rewards of Best Misclassified Image in TS-QL Averaged over 10 Folds', fontsize=8)
    axes[0, 1].set_xlabel('Episodes', fontsize=8)
    axes[0, 1].set_ylabel('Cumulative Rewards', fontsize=8)
    axes[0, 1].plot(episodes, average_cum_r_TS_QL_list_one_shot, color='red', label='oneshot e-decay')
    axes[0, 1].fill_between(episodes, average_cum_r_TS_QL_list_one_shot - std_dev_cum_r_TS_QL_list_one_shot,
                            average_cum_r_TS_QL_list_one_shot + std_dev_cum_r_TS_QL_list_one_shot, color='red',
                            alpha=0.2, label='_nolegend_')
    axes[0, 1].plot(episodes, average_cum_r_TS_QL_list_harmonic, color='orange', label='harmonic e-decay')
    axes[0, 1].fill_between(episodes, average_cum_r_TS_QL_list_harmonic - std_dev_cum_r_TS_QL_list_harmonic,
                            average_cum_r_TS_QL_list_harmonic + std_dev_cum_r_TS_QL_list_harmonic, color='orange',
                            alpha=0.2, label='_nolegend_')
    axes[0, 1].plot(episodes, average_cum_r_TS_QL_list_random, color='blue', label='random selection')
    axes[0, 1].fill_between(episodes, average_cum_r_TS_QL_list_random - std_dev_cum_r_TS_QL_list_random,
                            average_cum_r_TS_QL_list_random + std_dev_cum_r_TS_QL_list_random, color='blue', alpha=0.2,
                            label='_nolegend_')
    axes[0, 1].set_yticks(y_ticks_cum_r)
    axes[0, 1].set_yticklabels(y_ticks_cum_r)

    axes[1, 0].set_title('(c) Max Q-Values of Best Misclassified Image in TS-QL-HF Averaged over 10 Folds', fontsize=8)
    axes[1, 0].set_xlabel('Episodes', fontsize=8)
    axes[1, 0].set_ylabel('Max Q-values', fontsize=8)
    axes[1, 0].plot(episodes, average_max_Q_values_TS_QL_HF_list_one_shot, color='red', label='oneshot RBED')
    axes[1, 0].fill_between(episodes,
                            average_max_Q_values_TS_QL_HF_list_one_shot - std_dev_max_Q_values_TS_QL_HF_list_one_shot,
                            average_max_Q_values_TS_QL_HF_list_one_shot + std_dev_max_Q_values_TS_QL_HF_list_one_shot,
                            color='red', alpha=0.2, label='_nolegend_')
    axes[1, 0].plot(episodes, average_max_Q_values_TS_QL_HF_list_harmonic, color='orange', label='harmonic RBED')
    axes[1, 0].fill_between(episodes,
                            average_max_Q_values_TS_QL_HF_list_harmonic - std_dev_max_Q_values_TS_QL_HF_list_harmonic,
                            average_max_Q_values_TS_QL_HF_list_harmonic + std_dev_max_Q_values_TS_QL_HF_list_harmonic,
                            color='orange', alpha=0.2, label='_nolegend_')
    axes[1, 0].plot(episodes, average_max_Q_values_TS_QL_HF_list_random, color='blue', label='random selection')
    axes[1, 0].fill_between(episodes,
                            average_max_Q_values_TS_QL_HF_list_random - std_dev_max_Q_values_TS_QL_HF_list_random,
                            average_max_Q_values_TS_QL_HF_list_random + std_dev_max_Q_values_TS_QL_HF_list_random,
                            color='blue', alpha=0.2, label='_nolegend_')

    axes[1, 1].set_title('(d) Cumulative Rewards of Best Misclassified Image in TS-QL-HF Averaged over 10 Folds', fontsize=8)
    axes[1, 1].set_xlabel('Episodes', fontsize=8)
    axes[1, 1].set_ylabel('Cumulative Rewards', fontsize=8)
    axes[1, 1].plot(episodes, average_cum_r_TS_QL_HF_list_one_shot, color='red', label='oneshot RBED')
    axes[1, 1].fill_between(episodes, average_cum_r_TS_QL_HF_list_one_shot - std_dev_cum_r_TS_QL_HF_list_one_shot,
                            average_cum_r_TS_QL_HF_list_one_shot + std_dev_cum_r_TS_QL_HF_list_one_shot, color='red',
                            alpha=0.2, label='_nolegend_')
    axes[1, 1].plot(episodes, average_cum_r_TS_QL_HF_list_harmonic, color='orange', label='harmonic RBED')
    axes[1, 1].fill_between(episodes, average_cum_r_TS_QL_HF_list_harmonic - std_dev_cum_r_TS_QL_HF_list_harmonic,
                            average_cum_r_TS_QL_HF_list_harmonic + std_dev_cum_r_TS_QL_HF_list_harmonic, color='orange',
                            alpha=0.2, label='_nolegend_')
    axes[1, 1].plot(episodes, average_cum_r_TS_QL_HF_list_random, color='blue', label='random selection')
    axes[1, 1].fill_between(episodes, average_cum_r_TS_QL_HF_list_random - std_dev_cum_r_TS_QL_HF_list_random,
                            average_cum_r_TS_QL_HF_list_random + std_dev_cum_r_TS_QL_HF_list_random, color='blue',
                            alpha=0.2, label='_nolegend_')
    axes[1, 1].set_yticks(y_ticks_cum_r)
    axes[1, 1].set_yticklabels(y_ticks_cum_r)

    plt.figlegend(['one-shot RBED','harmonic RBED', 'random selection'], loc='lower center', fontsize=8)
    plt.tight_layout()
    plt.show()

def plotDoubleQLearningPerformanceComparison(episodes,
                                             average_max_Q_values_TS_QL, std_dev_max_Q_values_TS_QL, average_cum_r_TS_QL, std_dev_cum_r_TS_QL,
                                             average_max_Q_values_TS_DQL, std_dev__max_Q_values_TS_DQL, average_cum_r_TS_DQL, std_dev_cum_r_TS_DQL,
                                             average_max_Q_values_TS_QL_HF, std_dev_max_Q_values_TS_QL_HF, average_cum_r_TS_QL_HF, std_dev_cum_r_TS_QL_HF,
                                             average_max_Q_values_TS_DQL_HF, std_dev_max_Q_values_TS_DQL_HF, average_cum_r_TS_DQL_HF, std_dev_cum_r_TS_DQL_HF
                                             ):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axes[0, 0].set_title('(a) Max Q-Values of Best Misclassified Image in TS-QL Averaged over 10 Folds', fontsize=8)




def plot_actions_stats(dataLoader, networkName, actions, stats, filename):
    plt.bar(actions, height=stats)
    plt.title('actions statistics')
    plt.ylabel('number of times action was chosen')
    plt.xlabel('action name')
    plt.savefig(os.path.join(dataLoader.resultsDir, 'actions_stats' + filename + networkName + dataLoader.datasetInfo +
                             '.png'))
    plt.show()


def plot_conf_matrix(dataLoader, networkName, conf_matrix, classes, filename):
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(dataLoader.resultsDir, 'conf_matrix' + filename + networkName + dataLoader.datasetInfo +
                             '.png'))

def print_classification_details(statController):
    print("accuracy: ", statController.accuracy)
    print("precision: ", statController.precision)
    print("recall: ", statController.recall)
    print("F1 score: ", statController.f1Score)
    print("report: ", statController.report)
