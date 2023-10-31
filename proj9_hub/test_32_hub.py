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
        
        samples = np.random.normal(0, 0.1, (trn_sizes.batchs-1)*trn_sizes.cepsTone) 
        samplestf = tf.convert_to_tensor(samples, dtype=tf.float32)
        vadLabel3 = np.random.randint(0, 1, 250, dtype=np.int32)
        vadLabelTf = tf.convert_to_tensor(vadLabel3, dtype=tf.float32)   
                
        return {samplestf, vadLabelTf}  
    
    def getBatch4(self, trn_sizes):  
        
        samples = np.random.normal(0, 0.1, (trn_sizes.batches-1)*trn_sizes.cepsTone) 
        mfccs = tf.convert_to_tensor(samples, dtype=tf.float32)
        mfccs3 = tf.reshape(mfccs, [trn_sizes.batches-1, 1, trn_sizes.cepsTone], )
        vadLabel3 = np.random.randint(2, size =(249, 1), dtype=np.int32)
        #vadLabel3 = np.random.randint(0, 1, 249, dtype=np.int32)
        vadLabelTf = tf.convert_to_tensor(vadLabel3, dtype=tf.float32)   
#        vadLabelTf = tf.reshape(vadLabelOut[:-1 ],[trn_sizes.batches-1, 1])        
        return {mfccs3, vadLabelTf}  
    

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
    vadModel = create_model(trn_sizes)
    #vadModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    vadModel.build(input_shape = [None, 10, trn_sizes.cepsTone])
        
    #vadModel = tf.keras.models.load_model('/content/best_on_batch_bug/tree/main/proj9_hub/saved_model/my_model_31.keras')
   
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
        mfccs3, vadLabel = vad.getBatch4(trn_sizes)
        print('mfccs3 value = ', tf.keras.backend.eval(mfccs3[1,:]) ) 
        #print('vadLabel value = ', tf.keras.backend.eval(vadLabel[0:3]) )              
        #logs = vadModel.train_on_batch(mfccs3, vadLabel)
        logs = vadModel.test_on_batch(mfccs3, vadLabel) 
        #logs = vadModel.predict_on_batch(mfccs3) 
        print('string logs = ', str(logs))
        #print('string logs = ', str(logs[0:3]))  # for printing prediction result
        #tensorboard.on_batch_end(i, named_logs(vadModel, [logs]))
        
    #vadModel.save("saved_model/my_model_32.keras");         
    #tensorboard.on_test_end(None)
            
    print("test is done!")
        

main(FLAGS)
    

