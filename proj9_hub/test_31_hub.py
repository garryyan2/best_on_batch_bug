import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow as tf
import tensorboard
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense
import random

tf.compat.v1.disable_eager_execution()



class trn_sizes:
    fm_length = 256
    fm_step = 128
    LSTMCell = 20
    timeStep = 1   # is 1 only for now
    batches = 250 
    batch_samples = batches*timeStep*fm_step
    noise_insert_time = 2 # second
    noise_chi2_df = 10
    sampling_rate = 16000
    time_per_batch = batches*timeStep*fm_step/sampling_rate   # 2s for 250 batch size
    noise_chi2_scale = noise_insert_time*sampling_rate/noise_chi2_df
    cepsTone = 12

trn_sizes = trn_sizes()

class inputBatch(object, ):

    
    def __init__(self, trn_sizes):
        self.state = 0 # 
        self.iirPinkSt = [0.0, 0.0, 0.0]
        self.pinkCount = 0
        self.pinkLeft = 0
        self.pinkBlkCnt = 0
        self.noisePower = 0.02
        self.iirPinkB1 = [0.010000000000000,  -0.018940429700000,   0.009585641562817,  -0.000621320035262]
        self.iirPinkA1 = [1.000000000000000,  -2.479309080000000 ,  1.985012853639686,  -0.505600430025288]
        self.remainingSamples = 0
        self.labelIdx = 0      # index based on the batch 
        self.samplesToBeFill = trn_sizes.batch_samples
        self.vadSampleIdx = 0  # index based on the audio file
        self.vadLabel = np.zeros(trn_sizes.batches)
        self.samplesOut = np.zeros(trn_sizes.batch_samples, dtype=float)
        self.indexa = 0

    def getBatch3(self, trn_sizes):  
        
        samples = np.random.normal(0, 0.1, trn_sizes.batch_samples) 
        samplestf = tf.convert_to_tensor(samples, dtype=tf.float32)
        vadLabel3 = np.random.randint(0, 1, 250, dtype=np.int32)
        vadLabelTf = tf.convert_to_tensor(vadLabel3, dtype=tf.float32)   
                
        return {samplestf, vadLabelTf}  
    

def create_model(trn_sizes):

    from tensorflow.keras import layers

    vadModel = tf.keras.Sequential()
    vadModel.add(layers.LSTM(trn_sizes.LSTMCell))
    vadModel.add(Dense(10, activation='sigmoid'))
    vadModel.add(Dense(1, activation='sigmoid'))
    vadModel.compile(optimizer='adam', loss='mse')
    return vadModel

FLAGS = None

def named_logs(model, logs):
    
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]

    return result   
    


def main(FLAGS):

    print('tensorflow version: ' +  tf.__version__)
    print(tf.version.GIT_VERSION, tf.version.VERSION)
    dirName = os.path.dirname(os.path.abspath(__file__)) +'/MSLB0'
                    
    random.seed(11)
 
    #vad = input_data.inputBatch(trn_sizes)     
    vad = inputBatch(trn_sizes)  
    #vadModel = create_model(trn_sizes)
    #vadModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #vadModel.build(input_shape = [None, 10, trn_sizes.cepsTone])
        
    vadModel = tf.keras.models.load_model('saved_model/my_model_31.keras')
    
    print('model length = ', len(vadModel.weights))
    
    print(vadModel.summary())
    #plot_model(vadModel, to_file = "logs/scalars/" + 'model108.png')

    #tf.keras.utils.plot_model(vadModel, to_file = 'my_5th_model.png')
   
    
 
    
    batch_size=trn_sizes.batches;
    
    tensorboard = keras.callbacks.TensorBoard(
      #log_dir='/tmp/my_tf_logs',
      log_dir= "logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S"),
      histogram_freq=0,
      batch_size=batch_size,
      write_graph=True,
      write_grads=True,
      update_freq="batch"
    )
    tensorboard.set_model(vadModel)   
    
 
         
    for i in range(3):  
        samplesOut, vadLabelOut = vad.getBatch3(trn_sizes)
        
        print('index = ', i)
        stfts = tf.signal.stft(samplesOut, frame_length=trn_sizes.fm_length, frame_step=trn_sizes.fm_step,
                               fft_length=trn_sizes.fm_length,  
                               window_fn=tf.signal.hann_window, pad_end=False)
        spectrograms = tf.abs(stfts)  
        
         
        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 7600.0, 64
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, trn_sizes.sampling_rate, lower_edge_hertz,
                upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
                spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
        
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
                log_mel_spectrograms)[..., :trn_sizes.cepsTone] 
                           
             
        vadLabel = tf.reshape(vadLabelOut[:-1 ],[trn_sizes.batches-1, 1]) 

 
        mfccs3 = tf.reshape(mfccs, [trn_sizes.batches-1, 1, trn_sizes.cepsTone], )
    
        print('mfccs3 value = ', tf.keras.backend.eval(mfccs3[1,:]) )               
        #logs = vadModel.train_on_batch(mfccs3, vadLabel)
        logs = vadModel.test_on_batch(mfccs3, vadLabel) 
        print('string logs = ', str(logs))
        tensorboard.on_batch_end(i, named_logs(vadModel, [logs]))
     
    #print("save model!")
    #vadModel.save("saved_model/my_model_tmp.keras");  #4 is fine for now
    #tf.saved_model(vadModel, "saved_model/my_model3")
            
    tensorboard.on_test_end(None)
            
    print("test is done!")
            
main(FLAGS)
    

