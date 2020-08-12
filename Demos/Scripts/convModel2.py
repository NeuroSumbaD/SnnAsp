import nengo
import numpy as np
import glob
from scipy.io import wavfile

dt = 0.001 #Fixed for nengo-gui

soundList = glob.glob("./Dataset/LibriSpeech/dev-clean-mix/*.wav")

sampRate, sound = wavfile.read(soundList[0])
sound = sound/np.max(sound) #normalize and convert to float
    
def bufferSound(t, buffSize=240, presDur=240, sound=sound):
    '''Rotates through wave data but in small chunks or buffers
        and with each chunk displayed for presDur'''
    n = t/dt/presDur
    duration = sound.size
    wrappedIndex = int(n % (duration-buffSize))
    return sound[wrappedIndex*buffSize:(wrappedIndex+1)*buffSize]
    


with nengo.Network() as model:
    # Input node representing the wavform
    inProbe = nengo.Node(lambda t: bufferSound(t), size_out=240, label="sound")
    
    # Ensemble weights are precomputed by the Nengo library
    # when defined this way
    spikeGen = nengo.networks.EnsembleArray(n_neurons=240*3, n_ensembles=240, label="spike sound")
    nengo.Connection(inProbe, spikeGen.input)
    
    # Define convolution transform
    conv = nengo.Convolution(4, input_shape=(1,240), kernel_size=(5,),
        strides=(1,), padding="same")
    # Define convolution layer
    convLayer = nengo.Ensemble(conv.output_shape.size, 4)
    nengo.Connection(spikeGen.output, convLayer.neurons, transform=conv)
    # note: in nengo, convolutions operate across ensemble dimensions
    #not across time. In this case the 50 dimension will refer to the
    #neurons in spikeGen


#Start Nengo GUI
import nengo_gui
nengo_gui.GUI(__file__).start()