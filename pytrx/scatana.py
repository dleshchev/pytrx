

import numpy as np
import matplotlib.pyplot as plt
from pytrx.scatdata import ScatData
from pytrx import scatsim

class ScatAna(ScatData):

    def __init__(self, input):
        super().__init__(input, smallLoad=True)




