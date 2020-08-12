import nengo
import numpy as np
import glob
from scipy.io import wavfile

dt = 0.0001 #Fixed for nengo-gui

soundList = glob.glob("./Dataset/LibriSpeech/dev-clean-mix/*.wav")

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
    inProbe = nengo.Node(lambda t: repeatSound(t*1000), label="sound")
    
    # Ensemble weights are precomputed by the Nengo library
    #when defined this way
    spikeGen = nengo.Ensemble(50,1, label="spike sound")
    nengo.Connection(inProbe, spikeGen)
    
    # Define convolution transform
    conv = nengo.Convolution(4, input_shape=(1,50), kernel_size=(5,),
        strides=(1,), padding="same", init=init)
    # Apply to convolution layer
    convLayer = nengo.Ensemble(conv.output_shape.size, 4)
    nengo.Connection(spikeGen.neurons, convLayer.neurons, transform=conv)
    # note: in nengo, convolutions operate across ensemble dimensions
    #not across time. In this case the 50 dimension will refer to the
    #neurons in spikeGen

#Start Nengo GUI
import nengo_gui
nengo_gui.GUI(__file__).start()