from aiida.engine import WorkChain, ToContext
from aiida.orm import List

from aiida_n2p2.calculations.scaling import nnpScaling
from aiida_n2p2.calculations.train import nnpTraining
from aiida_n2p2.calculations.predict import nnpPredict


class MakeNNPWorkchain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("nbin",valid_type=Int)
        spec.input(
            "inputData", valid_type=SinglefileData, help="Training set"
        )
        spec.input(
            "inputNN", valid_type=SinglefileData, help="Test set"
        )

        spec.outline(
            cls.scaling
            cls.train
            cls.predict
        )
        spec.output('weights')
        

