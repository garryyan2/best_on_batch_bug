import math
import os.path
import random
import struct
from scipy import signal
from scipy.stats import chi2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

try:
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
    frontend_op = None
    
from tensorflow import keras    


'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if (fullPath.endswith(".WAV") or fullPath.endswith(".wav")):
                allFiles.append(fullPath)
                
    return allFiles



class inputBatch(object):

    
    def __init__(self, trn_sizes):
        self.state = 0 # 
        self.iirPinkSt = [0.0, 0.0, 0.0]
        self.pinkCount = 0
        self.pinkLeft = 0
        self.pinkBlkCnt = 0
        self.noisePower = 0.02
        # a 3 poles and 3 zeros filter for pink noise
        # c = [0.99572754 0.94790649 0.53567505]
        # d = [0.98443604 0.83392334 0.07568359]
        #self.iirPinkB1 = [1.000000000000000  -1.894042970000000   0.958564156281748  -0.062132003526167]
        #scaled version. The gain needs to be adjusted
        self.iirPinkB1 = [0.010000000000000,  -0.018940429700000,   0.009585641562817,  -0.000621320035262]
        self.iirPinkA1 = [1.000000000000000,  -2.479309080000000 ,  1.985012853639686,  -0.505600430025288]
        self.remainingSamples = 0
        self.labelIdx = 0      # index based on the batch 
        self.samplesToBeFill = trn_sizes.batch_samples
        self.vadSampleIdx = 0  # index based on the audio file
        self.vadLabel = np.zeros(trn_sizes.batches)
        self.samplesOut = np.zeros(trn_sizes.batch_samples, dtype=float)
        self.indexa = 0

        

    def Read_Phn_File(self, filename):
        """Loads an phn file and returns a float array of vads for each data frame.
    
        Args:
            filename: Path to the .phn file to load.
    
        Returns:
            Numpy array holding the vads for the training frame.
        """
        
        #get file object

        phn_file = filename[:-8]
        phn_file = phn_file + ".PHN"
        f = open(phn_file, "r")    
        h_count = 0
    
        while(True):
            #read next line

            line = f.readline()
            #if line is empty, you are done with all lines in the file
            if not line:
                break
            #you can access the line
            #if "h#" in line.endswith("h#"):
            if "h#" in line:
                #print("line-----------------++++++", line)
                line1 = line[:-3]
                line2 = line1.split()
                start1 = int(line2[0])
                end1 = int(line2[1])
                if h_count == 0:
                    vad_start = end1
                
                elif h_count == 1:
                    vad_end = start1
                    vad_length = end1
                    
                h_count = h_count + 1
                            
       
        #close file
        f.close
        #print("woo, st_end------------------------------")

        
        return {vad_start, vad_end, vad_length}

    
        
    
    def getBatch(self, listOfFiles, trn_sizes):  
    
        samples = []
        #vad_label = np.zeros(trn_sizes.batches)
        self.samplesToBeFilled = trn_sizes.batch_samples
       
        indexa = 0
        
        #print('listOfFiles count =', len(listOfFiles))
        for filename in listOfFiles:            
                
            vad_start, vad_end, vad_samples = self.Read_Phn_File(filename)
            #use sample_count instead of vad_samples
            #print('filename =', filename)
            
        
            self.vadSampleIdx = 0
           
            wav_file = open(filename, 'rb')

            # Main Header
            chunk_id = wav_file.read(4)
            assert chunk_id == b'RIFF', 'RIFF little endian, RIFX big endian: assume RIFF'
    
            chunk_size = struct.unpack('<I', wav_file.read(4))[0]
    
            wav_format = wav_file.read(4)
            assert wav_format == b'WAVE', wav_format
    
            # Sub Chunk 1
            sub_chunk_1_id = wav_file.read(4)
            assert sub_chunk_1_id == b'fmt ', sub_chunk_1_id
    
            sub_chunk_1_size = struct.unpack('<I', wav_file.read(4))[0]
    
            audio_format = struct.unpack('<H', wav_file.read(2))[0]
            assert audio_format == 1, '1 == PCM Format: assumed PCM'
    
            num_channels = struct.unpack('<H', wav_file.read(2))[0]
            assert num_channels == 1, '1 == Mono, 2 == Stereo: assumed Mono'
    
            sample_rate = struct.unpack('<I', wav_file.read(4))[0]
            assert sample_rate == 16000, 'assumed 16000'
    
            byte_rate = struct.unpack('<I', wav_file.read(4))[0]
            assert byte_rate == 32000, byte_rate
    
            # Could this be something other than an int?
            block_align = struct.unpack('<H', wav_file.read(2))[0]
            assert block_align == 2, block_align
    
            bits_per_sample = struct.unpack('<H', wav_file.read(2))[0]
            assert bits_per_sample == 16, bits_per_sample
    
            # Sub Chunk 2
            sub_chunk_2_id = wav_file.read(4)
            assert sub_chunk_2_id == b'data', sub_chunk_2_id
    
            sub_chunk_2_size = struct.unpack('<I', wav_file.read(4))[0]
    
        
            bytes_per_sample = bits_per_sample / 8
            assert (sub_chunk_2_size % bytes_per_sample) == 0, 'Uneven sample size'
    
            sample_count = int(sub_chunk_2_size / bytes_per_sample)
            #print('sample count =', sample_count)
            
            self.remainingSamples = sample_count
            self.vadSampleIdx = 0
            samples_tmp = []
            
            while self.remainingSamples >= self.samplesToBeFilled:
                print('tobefilled 1=', self.samplesToBeFilled)
                              
                for _ in range(self.samplesToBeFilled):
                    samples_tmp.append((struct.unpack('<h', wav_file.read(2))[0])/32786.0)
                
                #samplesf = samples_tmp
                samples.extend(samples_tmp)
                samples_tmp = []
                
                self.remainingSamples -= self.samplesToBeFilled
                samplestf = tf.convert_to_tensor(samples, dtype=tf.float32)
                
                    
                
                
            
                for  _ in range(round(self.samplesToBeFilled/trn_sizes.fm_step)):                    
                    self.vadSampleIdx += trn_sizes.fm_step
                    
                    if (self.vadSampleIdx > vad_start) and (self.vadSampleIdx < (vad_end + 0.7*trn_sizes.fm_step)):                        
                        self.vadLabel[self.labelIdx] = 1
                    self.labelIdx += 1
                    if self.labelIdx >= trn_sizes.batches:
                        self.labelIdx -=1
                
                vadLabelTf = tf.convert_to_tensor(self.vadLabel, dtype=tf.float32)
                 
                self.samplesOut = np.array(samples) 
                                            
                assert len(self.samplesOut) == trn_sizes.batch_samples, f"sample size is not correct"
                indexa += 1
                self.indexa += 1 
                yield {samplestf, vadLabelTf}      
                
                self.samplesToBeFilled = trn_sizes.batch_samples                    
                self.vadLabel = np.zeros(trn_sizes.batches)  
                samples = []
                self.labelIdx = 0
                      
            self.pinkLeft = round(chi2.rvs(trn_sizes.noise_chi2_df)*trn_sizes.noise_chi2_scale) 
            if self.remainingSamples > 0:
                for _ in range(self.remainingSamples):
                    samples_tmp.append((struct.unpack('<h', wav_file.read(2))[0])/32786.0)
                                    
                wav_file.close()  
                 
                samples.extend(samples_tmp)
                samples_tmp = []
                                   
                for  _ in range(round(self.remainingSamples/trn_sizes.fm_step)):                    
                    self.vadSampleIdx += trn_sizes.fm_step
                    
                    if self.vadSampleIdx > vad_start & self.vadSampleIdx < vad_end + 0.7*trn_sizes.fm_step:
                        self.vadLabel[self.labelIdx] = 1
                    self.labelIdx += 1
                    if self.labelIdx >= trn_sizes.batches:
                        self.labelIdx -=1
                    
                
                self.samplesToBeFilled -= self.remainingSamples
                                
            else:
                wav_file.close()   
            
              
            # filling noise
            while  self.pinkLeft >= self.samplesToBeFilled:
                whiteSamples = np.random.normal(0, self.noisePower, self.samplesToBeFilled) 

                pinkSamples, self.iirPinkSt = signal.lfilter(self.iirPinkB1, self.iirPinkA1, whiteSamples, -1, self.iirPinkSt) 
            
                samples.extend(pinkSamples) 
                
                self.labelIdx += round(self.samplesToBeFilled/trn_sizes.fm_step)
                if self.labelIdx >= trn_sizes.batches:
                    self.labelIdx = trn_sizes.batches - 1
                        
                self.pinkLeft -= self.samplesToBeFilled
                 
            
                
                samplestf = tf.convert_to_tensor(samples, dtype=tf.float32)
                vadLabelTf = tf.convert_to_tensor(self.vadLabel, dtype=tf.float32)   
                
                self.samplesOut = np.array(samples) 
                assert len(self.samplesOut) == trn_sizes.batch_samples, f"sample size is not correct" 
                indexa += 1
                self.indexa += 1
                yield {samplestf, vadLabelTf}  
                
                self.vadLabel = np.zeros(trn_sizes.batches)  
                samples = [] 
                self.labelIdx = 0
                
                self.samplesToBeFilled = trn_sizes.batch_samples
                    

            if self.pinkLeft > 0:   
                whiteSamples = np.random.normal(0, self.noisePower, self.pinkLeft) 
            
                pinkSamples, self.iirPinkSt = signal.lfilter(self.iirPinkB1, self.iirPinkA1, whiteSamples, -1, self.iirPinkSt) 
                
                samples.extend(pinkSamples) 
                self.labelIdx += round(self.pinkLeft/trn_sizes.fm_step)
                if self.labelIdx >= trn_sizes.batches:
                    self.labelIdx = trn_sizes.batches - 1
   
                
                self.samplesToBeFilled -= self.pinkLeft
                            
                 

def load_phn_file2(filename, fm_length, fm_step, sampling_rate, audio_size):
    """Loads an phn file and returns a float array of vads for each data frame.

    Args:
        filename: Path to the .phn file to load.

    Returns:
        Numpy array holding the vads for the training frame.
    """
    

    f = open(filename, "r")
    
    
    h_count = 0
    

    while(True):
        #read next line
        line = f.readline()
        #if line is empty, you are done with all lines in the file
        if not line:
            break
        #you can access the line
        #if "h#" in line.endswith("h#"):
        if "h#" in line:
            print("line++++++++++++++++++++", line)
            line1 = line[:-3]
            line2 = line1.split()
            start1 = math.ceil(int(line2[0])/fm_step)
            end1 = math.ceil(int(line2[1])/fm_step)
            if h_count == 0:
                vad_start = end1
            
            if h_count == 1:
                vad_end = start1
                vad_length = end1
                
            h_count = h_count + 1
            
    vad_start = vad_start - math.floor(fm_length/fm_step/2)
    vad_end = vad_end - math.floor(fm_length/fm_step/2)          
    vad = np.zeros([vad_length, 1], np.bool_)    
    vad[vad_start:vad_end] = True
            
        
    
    #close file
    f.close
    print("woo, st_end------------------------------")
    vad_tf = tf.convert_to_tensor(vad, np.bool_)
    
    return vad_tf


    
    

    
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytesOld, timestamp, duration):
        self.bytesOld = bytesOld
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    """
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n  
    """
    
    return Frame(audio[offset:offset + n], timestamp, duration)


