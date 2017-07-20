"""Testing for the VotingClassifier"""

from __future__ import print_function

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.testing import assert_equal, assert_true, assert_false
from sklearn.utils.testing import assert_raise_message

from imblearn.ensemble import EasyEnsembleGeneralization as EEG

RND_SEED = 0
X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
              [1.25192108, -0.22367336], [0.53366841, -0.30312976],
              [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
              [0.83680821, 1.72827342], [0.3084254, 0.33299982],
              [0.70472253, -0.73309052], [0.28893132, -0.38761769],
              [1.15514042, 0.0129463], [0.88407872, 0.35454207],
              [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
              [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
              [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
              [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])


def test_estimator_init():

    eeg = EEG(n_estimators=0)
    msg = "n_estimators must be greater than zero, got 0."
    assert_raise_message(ValueError, msg, eeg.fit, X, y)


def test_predict_proba_hardvoting():
    eeg = EEG(voting='hard', random_state=RND_SEED).fit(X, y)
    msg = "predict_proba is not available when voting='hard'"
    assert_raise_message(AttributeError, msg, eeg.predict_proba, X)


def test_notfitted():
    eeg = EEG()
    msg = ("This EasyEnsembleGeneralization instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg, eeg.predict_proba, X)


def test_majority_label():
    """Check classification by majority vote."""
    eeg = EEG(voting='soft', random_state=RND_SEED)
    scores = cross_val_score(eeg, X, y, cv=5, scoring='roc_auc')
    print(scores.mean())
    assert_almost_equal(scores.mean(), 0.65, decimal=2)


def test_predict_on_toy_problem():
    """Manually check predicted class labels for the toy dataset."""
    eeg = EEG(voting='hard', random_state=RND_SEED)
    assert_equal(all(eeg.fit(X, y).predict(X[0:6])), all([0, 1, 0, 0, 0, 1]))


def test_gridsearch():
    """Check GridSearch support."""
    eeg = EEG(random_state=RND_SEED)

    params = {'voting': ['soft', 'hard'],
              'n_estimators': [2, 3, 4]}

    grid = GridSearchCV(estimator=eeg, param_grid=params, cv=3)
    grid.fit(X, y)


def test_parallel_predict():
    """Check parallel backend of EasyEnsembleGeneralization on the toy dataset."""
    eeg1 = EEG(voting='soft', random_state=RND_SEED, n_jobs=1).fit(X, y)
    eeg2 = EEG(voting='soft', random_state=RND_SEED, n_jobs=2).fit(X, y)

    assert_array_equal(eeg1.predict(X), eeg2.predict(X))
    assert_array_equal(eeg1.predict_proba(X), eeg2.predict_proba(X))
