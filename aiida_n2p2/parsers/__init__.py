"""AiiDA parser plugins for n2p2."""

from aiida_n2p2.parsers.predict import nnpPredictParser
from aiida_n2p2.parsers.scaling import nnpScaleParser
from aiida_n2p2.parsers.train import nnpTrainParser

__all__ = ('nnpPredictParser', 'nnpScaleParser', 'nnpTrainParser')
