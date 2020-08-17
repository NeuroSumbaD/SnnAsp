from abc import ABC, abstractmethod

from .. import pipeline
from . import keras
import nengo

class PureNef(keras.Net):
    pass

class Mixed(keras.Net):
    pass
