import numpy as np
import cv2

from random import randrange
from scipy.ndimage.interpolation import rotate


class QLearningModel:
    def __init__(self):
        self.alpha = 1
        self.gamma = 0
        self.angle1 = 90
        self.angle2 = 180
        self.angle3 = 12.5
        self.angle4 = -12.5
        self.M_translation = np.float32([[1, 0, 15], [0, 1, 15]])
        self.M_translation_2 = np.float32([[1, 0, -15], [0, 1, -15]])

        self.action = None
        self.actions = dict([(0, self.action_rotate_1), (1, self.action_rotate_2), (2, self.diagonal_translation)])

        self.r_h = 0
        self.r = 0
        self.episode = 0
        self.maxIter = 10
        self.rewards = []
        self.cum_rewards = []
        self.max_q_estimates = []
        self.cum_rewards_all_images = []
        self.max_q_estimates_all_images = []

        self.states = [0, 1]
        self.total = [0, 0, 0]
        self.tableQ = np.zeros((len(self.states), len(self.actions)))

    def action_rotate_1(self, picture):
        return rotate(picture, self.angle1, reshape=False)

    def action_rotate_2(self, picture):
        return rotate(picture, self.angle2, reshape=False)

    def action_rotate_3(self, picture):
        return rotate(picture, self.angle3, reshape=False)

    def action_rotate_4(self, picture):
        return rotate(picture, self.angle4, reshape=False)

    def diagonal_translation(self, picture):
        rows, cols = picture.shape[:2]
        translated = cv2.warpAffine(picture, self.M_translation, (cols, rows))
        return np.array(translated)

    def diagonal_translation_2(self, picture):
        rows, cols = picture.shape[:2]
        translated = cv2.warpAffine(picture, self.M_translation_2, (cols, rows))
        return np.array(translated)

    def selectAction(self):
        return randrange(len(self.actions))

    def epsilon_greedy_selection(self, eps):
        p = np.random.random()
        if p < eps:
            rand_action = np.random.choice(len(self.actions))
            # print(f'Rand action: {rand_action}')
            return int(rand_action)
        else:
            total = np.sum(self.tableQ, axis=0)
            # print(f'Total: {total}')
            best_action = np.argmax(total)
            # print(f'Best action: {best_action}')
            return int(best_action)

    def apply_action(self, action, img):
        return self.actions[action](img)

    def get_features_metric(self, features):
        return np.std(features)

    def get_reward(self, m1, m2):
        return 1 if m2 > m1 else -1

    def define_state(self, reward):
        # print("State chosen: " + str(0 if reward > 0 else 1))
        return 0 if reward > 0 else 1

    def update_tableQ(self, state, action, reward):
        # print(
        #     f'{self.tableQ[state][action]} + {self.alpha} * ({reward} + {self.gamma} * {max(self.tableQ[state])}) - {self.tableQ[state][action]} ')
        self.tableQ[state][action] = self.tableQ[state][action] + (
                self.alpha * (reward + self.gamma * max(self.tableQ[state]) - self.tableQ[state][action]))
        # print(f'New Table Q(s,a) value: {self.tableQ[state][action]}')

    def get_best_max_Q_values_one_img(self):
        array_totals = [np.sum(arr) for arr in self.max_q_estimates_all_images]
        highest_total_index = np.argmax(array_totals)
        best_q_estimates_list = self.max_q_estimates_all_images[highest_total_index]

        return best_q_estimates_list

    def get_best_max_cum_r_one_img(self):
        array_totals = [np.sum(arr) for arr in self.cum_rewards_all_images]
        highest_total_index = np.argmax(array_totals)
        best_cum_r_list = self.cum_rewards_all_images[highest_total_index]

        return best_cum_r_list

    def perform_iterative_Q_learning(self, cnn, img, classes, action_selection_strategy, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

        print(f'selected strategy: {action_selection_strategy}')

        print('Reset here')
        self.tableQ = np.zeros((len(self.states), len(self.actions)))
        self.rewards = []
        self.cum_rewards = []
        self.max_q_estimates = []
        state = 0

        img_features = cnn.get_output_base_model(img)  # The output activation function of the last layer (original image)
        m1 = self.get_features_metric(img_features)  # The std deviation of img_features (original image)
        # print("m1: " + str(m1))

        # Make sure this is in every class
        # --------------------------------
        eps = 1.0
        decay_index = 0
        # ---------------------------------

        # Repeat for 60 iterations
        for i in range(self.maxIter):
            self.max_q_estimates.append(np.max(self.tableQ))
            # print(f'Max q value list updated: {str(self.max_q_estimates)}')

            print("i: " + str(i))
            print('eps: ' + str(eps))
            self.episode += 1

            if action_selection_strategy == 'random':
                # ---------------------------------------------
                # Random strategy
                action = self.selectAction()
                # ---------------------------------------------
            elif action_selection_strategy == 'harmonic-sequence-e-decay':
                # ----------------------------------------------------
                # Epsilon strategy with harmonic sequence decay part 1
                action = self.epsilon_greedy_selection(eps)
                # ----------------------------------------------------
            elif action_selection_strategy == 'one-shot-e-decay':
                # ----------------------------------------------
                # Epsilon strategy with one shot decay part 1
                action = self.epsilon_greedy_selection(eps)
                # ----------------------------------------------

            # statsController.updateAllActionStats(action)  # Record the chosen action in the list
            modified_img = self.apply_action(action, img)  # Apply action chosen (Does not affect original image)

            modified_img_features = cnn.get_output_base_model(modified_img)  # The output activation function of the last layer (modified image)
            m2 = self.get_features_metric(modified_img_features)  # The std deviation of img_features (modified image)
            # print(m2)
            reward = self.get_reward(m1, m2)  # Calculate reward using m2-m1, (m2 > m1 for positive reward) (new std_dev must be higher for positive reward)
            self.rewards.append(reward)

            if action_selection_strategy == 'harmonic-sequence-e-decay':
                # ----------------------------------------------------
                # Epsilon strategy with harmonic sequence decay part 2
                if reward == 1:
                    eps = 1 / (decay_index + 1) ** 2
                    decay_index = decay_index + 1
                # ----------------------------------------------------
            elif action_selection_strategy == 'one-shot-e-decay':
                # --------------------------------------------
                # Epsilon strategy with one shot decay part 2
                if reward == 1:
                    eps = 0
                # --------------------------------------------

            state = self.define_state(reward)  # Choose a state in the Q-Table, state 0 is reward > 0
            self.update_tableQ(state, action, reward)  # Update the action value

            print("Action used: " + str(action))
            # print("Stddev of feature map of transformed image: " + str(m2))

            # print(pd.DataFrame(self.tableQ))

        self.cum_rewards = np.cumsum(self.rewards)
        self.cum_rewards_all_images.append(self.cum_rewards)
        self.max_q_estimates_all_images.append(self.max_q_estimates)

    def choose_optimal_action(self):
        # print(f'Optimal action: {np.where(self.tableQ == np.amax(self.tableQ))[1][0]}')
        return np.where(self.tableQ == np.amax(self.tableQ))[1][0]
