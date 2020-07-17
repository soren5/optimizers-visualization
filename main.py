from benchmark_functions.beale import Beale
from visualization.landscape_3d import landscape_3d

from tensorflow.keras.optimizers import Adam, SGD

beale = Beale()
landscape_3d(beale, [SGD(), Adam()])
