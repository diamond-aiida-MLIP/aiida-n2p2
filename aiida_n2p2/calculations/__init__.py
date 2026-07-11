"""AiiDA calculation plugins for n2p2."""

from aiida_n2p2.calculations.predict import nnpPredict
from aiida_n2p2.calculations.scaling import nnpScaling
from aiida_n2p2.calculations.train import nnpTraining

__all__ = ('nnpPredict', 'nnpScaling', 'nnpTraining')
