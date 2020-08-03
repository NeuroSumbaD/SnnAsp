import nengo
import numpy as np
import glob
from scipy.io import wavfile

dt = 0.0001 #Fixed for nengo-gui

soundList = glob.glob("../Dataset/LibriSpeech/dev-clean-mix/*.wav")

sampRate, sound = wavfile.read(soundList[0])
sound = sound/np.max(sound) #normalize and convert to float

init = np.reshape(np.sin(np.arange(1000)), (5,50,4))

def repeatSound(n, sound=sound):
    '''Rotates through wave data ad infinitum'''
    duration = sound.size
    wrappedIndex = int(n % duration)
    return sound[wrappedIndex]
    

with nengo.Network() as model:
    # Input node representing the wavform
    # inProbe = nengo.Node(lambda t: repeatSound(t*1000), label="sound")
    inProbe = nengo.Node(lambda t: np.sin(2*np.pi*t), label="sound")
    
    # Create an ensemble to convert to a spiking input format
    #that can be more easily passed to Loihi
    # Ensemble weights are precomputed by the Nengo library
    #when defined this way
    spikeGen = nengo.Ensemble(50,1, label="spike sound")
    nengo.Connection(inProbe, spikeGen)
    
    # Define convolution transform
    conv = nengo.Convolution(4, input_shape=(1,50), kernel_size=(5,),
        strides=(1,), padding="same", init=init)
    # Define convolution layer
    convLayer = nengo.Ensemble(conv.output_shape.size, 4)
    nengo.Connection(spikeGen.neurons, convLayer.neurons, transform=conv)