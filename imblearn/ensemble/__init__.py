"""
The :mod:`imblearn.ensemble` module include methods generating
under-sampled subsets combined inside an ensemble.
"""

from .easy_ensemble import EasyEnsemble
from .easy_ensemble_generalization import EasyEnsembleGeneralization
from .balance_cascade import BalanceCascade

__all__ = ['EasyEnsemble', 'EasyEnsembleGeneralization', 'BalanceCascade']
