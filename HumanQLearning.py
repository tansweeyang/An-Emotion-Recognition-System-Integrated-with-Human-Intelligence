import numpy as np
import cv2

from random import randrange
from scipy.ndimage.interpolation import rotate
from ImageHelper import NumpyImg2Tensor


class HumanQLearning:
    def __init__(self):
        self.alpha = 1 # 0.4
        self.gamma = 0 # 0.3
        self.angle1 = 90
        self.angle2 = 180
        self.angle3 = 12.5
        self.angle4 = -12.5
        self.M_translation = np.float32([[1, 0, 15], [0, 1, 15]])
        self.M_translation_2 = np.float32([[1, 0, -15], [0, 1, -15]])

        self.action = None
        self.rand_Q_table = 0

        self.actions = dict([(0, self.action_rotate_1), (1, self.action_rotate_2), (2, self.diagonal_translation)])

        self.max_q_estimates = []
        self.rewards = []
        self.cum_rewards = []
        self.cum_rewards_all_images = []
        self.max_q_estimates_all_images = []

        self.r_h = 0
        self.r = 0
        self.episode = 0
        self.maxIter = 10

        self.total = [0, 0, 0]

        # Initialize QA,QB,s
        self.states = [0, 1]
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
        print(f'selected strategy: {action_selection_strategy}')
        self.alpha = alpha
        self.gamma = gamma

        print(f'Reset here')
        self.tableQ = np.zeros((len(self.states), len(self.actions)))
        self.rewards = []
        self.cum_rewards = []
        self.max_q_estimates = []

        # Make sure this is in every class
        # --------------------------------
        eps = 1.0
        decay_index = 0
        # ---------------------------------

        # Collect feedback for all actions first
        # ---------------------------------------
        action_feedback = {}  # Store human feedback for each action

        for action in self.actions:
            modified_img = self.apply_action(action, img)

            probabilities_vector = cnn.model.predict(NumpyImg2Tensor(modified_img))
            prediction = np.argmax(probabilities_vector)
            emotion = next(key for key, val in classes.items() if val == prediction)
            print(f'Is {emotion} the correct emotion?')

            scale_percent = 200  # percent of original size
            width = int(modified_img.shape[1] * scale_percent / 100)
            height = int(modified_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            modified_img = cv2.resize(modified_img, dim, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('Modified image', modified_img)
            cv2.waitKey(0)

            print('\n3. Get human reward feedback')
            print('Enter one of the following options: (1) for correct prediction, (-1) for wrong prediction')
            self.r = input()
            self.r = int(self.r)
            action_feedback[action] = self.r

        # Use the collected feedback for subsequent iterations
        for i in range(self.maxIter):
            self.max_q_estimates.append(np.max(self.tableQ))
            print(f'Max q value list updated: {str(self.max_q_estimates)}')

            self.episode = self.episode + 1
            print(f'Episode: {self.episode}')
            print('eps: ' + str(eps))

            if action_selection_strategy == 'random':
                # ---------------------------------------------
                # Random strategy
                self.action = self.selectAction()
                # ---------------------------------------------
            elif action_selection_strategy == 'harmonic-sequence-e-decay':
                # ----------------------------------------------------
                # Epsilon strategy with harmonic sequence decay part 1
                self.action = self.epsilon_greedy_selection(eps)
                # ----------------------------------------------------
            elif action_selection_strategy == 'one-shot-e-decay':
                # ----------------------------------------------
                # Epsilon strategy with one shot decay part 1
                self.action = self.epsilon_greedy_selection(eps)
                # ----------------------------------------------

            modified_img = self.apply_action(self.action, img)
            self.r = action_feedback[self.action]

            state = self.define_state(self.r)
            self.update_tableQ(state, self.action, self.r)

            self.rewards.append(self.r)

            # print(f'max_q_estimates_all_images: {self.max_q_estimates_all_images}')

            if action_selection_strategy == 'harmonic-sequence-e-decay' and self.r == 1:
                eps = 1 / (decay_index + 1) ** 2
                decay_index = decay_index + 1
            elif action_selection_strategy == 'one-shot-e-decay' and self.r != 1:
                eps = 0

        self.cum_rewards = np.cumsum(self.rewards)
        self.cum_rewards_all_images.append(self.cum_rewards)
        self.max_q_estimates_all_images.append(self.max_q_estimates)

    def choose_optimal_action(self):
        total = np.sum(self.tableQ, axis=0)
        best_action = np.argmax(total)
        # print(f'Optimal action: {best_action}')
        return best_action
