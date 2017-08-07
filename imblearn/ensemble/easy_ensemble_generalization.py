"Easy Ensemble Generalization"

# Authors: Christos Aridas
#
# License: MIT
from __future__ import print_function

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble.base import BaseEnsemble, _set_random_states
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from ..pipeline import Pipeline
from ..under_sampling import RandomUnderSampler

MAX_INT = np.iinfo(np.int32).max


class EasyEnsembleGeneralization(BaseEnsemble, ClassifierMixin):
    """This classifier generalize the Easy Ensemble algorithm for imbalanced 
       datasets.

    Parameters
    ----------
    estimator : object or None, optional (default=None)
        Invoking the ``fit`` method on the ``EasyEnsembleGeneralization`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to `None` using
        ``set_params``.

    sampler: object or None, optional (default=None)
        Invoking the ``fit`` method on the ``EasyEnsembleGeneralization`` will fit clones
        of those original samplers.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for ``fit``.
        If -1, then the number of jobs is set to the number of cores.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted estimators.

    classes_ : array-like, shape = [n_predictions]
        The classes labels.

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.ensemble import EasyEnsembleGeneralization as EEG
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eeg = EEG(voting='soft', random_state=0)
    >>> eeg = eeg.fit(X,y)
    >>> print(eeg.predict(X))
    [1 1 1 2 2 2]
    >>>
    """

    def __init__(self,
                 base_estimator=None,
                 base_sampler=None,
                 n_estimators=5,
                 voting='soft',
                 random_state=None,
                 n_jobs=1):

        self.base_estimator = base_estimator
        self.base_sampler = base_sampler
        self.n_estimators = n_estimators
        self.voting = voting
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(EasyEnsembleGeneralization, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _validate_sampler(self):
        """Check the sampler and set the base_sampler_ attribute."""

        if self.base_sampler is not None:
            self.base_sampler_ = self.base_sampler
        else:
            self.base_sampler_ = RandomUnderSampler()

        if self.base_sampler_ is None:
            raise ValueError("base_sampler cannot be None")

    def fit(self, X, y, sample_weight=None):
        """Build an ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """

        
        check_classification_targets(y)
        
        random_state = check_random_state(self.random_state)

        self._validate_estimator()
        self._validate_sampler()

        random_state = check_random_state(self.random_state)

        if not hasattr(self.base_sampler, 'random_state'):
            ValueError('Base sampler must have a random_state parameter')

        steps = [('sampler', self.base_sampler_),
                 ('estimator', self.base_estimator_)]
        pipeline_template = Pipeline(steps)

        pipelines = []
        for i in enumerate(range(self.n_estimators)):
            pipeline = clone(pipeline_template)
            _set_random_states(pipeline, random_state)
            pipelines.append(pipeline)

        ensemble_members = [[str(i), pipeline]
                            for i, pipeline in enumerate(pipelines)]

        self._voting = VotingClassifier(ensemble_members,
                                        voting=self.voting,
                                        n_jobs=self.n_jobs)
        self._voting.fit(X, y)

        self.classes_ = self._voting.classes_
        self.estimators_ = [pipeline.named_steps['estimator']
                            for pipeline in self._voting.estimators_]

        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        check_is_fitted(self, "_voting")
        return self._voting.predict(X)

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for all samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        check_is_fitted(self, "_voting")
        return self._voting.predict_proba(X)
