from typing import Dict

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier

class ActionClassificationScore:

    def __init__(self):
        pass

    def __call__(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        results = {}
        linear_results = self.compute_linear_results(actions, vectors, actions_count)
        rbf_results = self.compute_rbf_results(actions, vectors, actions_count)
        poly_results = self.compute_poly_results(actions, vectors, actions_count)
        linear_ovo_results = self.compute_ovo_results(actions, vectors, actions_count)

        results = dict(results, **linear_results, **rbf_results, **poly_results, **linear_ovo_results)

        return results

    def compute_ovo_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = OneVsOneClassifier(svm.LinearSVC(random_state=0, max_iter=10000))

        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results["linear_ovo/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"linear_ovo/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results

    def compute_linear_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = svm.LinearSVC(max_iter=10000)
        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results["linear/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"linear/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results

    def compute_rbf_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = svm.SVC(max_iter=10000)
        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results["rbf/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"rbf/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results

    def compute_poly_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = svm.SVC(kernel="poly", max_iter=10000)
        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results["poly/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"poly/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results

if __name__ == "__main__":

    actions = np.random.randint(0, 4, size=(1000))
    vectors = np.random.random([1000, 3])

    results = ActionClassificationScore()(actions, vectors, 5)

    print(results)