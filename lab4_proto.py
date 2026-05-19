
# DT2119, Lab 4 End-to-end Speech Recognition
import torch.nn as nn
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(n_mels=80),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35))
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = torchaudio.transforms.MelSpectrogram(n_mels=80)

# Functions to be implemented ----------------------------------

characters = "' abcdefghijklmnopqrstuvwxyz"
charact2index = {}
for index, char in enumerate(characters):
    charact2index[char] = index 
index2char = {index: char for char, index in  charact2index.items()}

def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    strings = ""
    for number in labels:
        strings+=index2char[number]
    return strings
        

def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    labels = []

    for char in text.lower():
        labels.append(charact2index[char])
    return labels

def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    print("Starting data processing:")
    for (waveform, _, utterance, _, _, _) in tqdm(data, desc="Processing batch"):
        spec = transform(waveform)
        spec = spec.squeeze(0).transpose(0, 1) # Rearrange the spectrogram tensor (1 channel, time first)
        spectrograms.append(spec)

        # Labels:
        #print("\nNew utterance!")
        #print(utterance)
        label = torch.tensor(strToInt(utterance)) # From stril to integer label
        labels.append(label)

        # Lengths (used in loss funct for masking):
        input_lengths.append(spec.shape[0] // 2) # Spectrogram reduces resolution to aproximately half (downsampling on first CNN layer) 
        label_lengths.append(label.shape[0])

    # Padding (nw is expecting all input have same length):
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True,padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True,padding_value=0)

    # Reshape
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)

    return spectrograms, labels, input_lengths, label_lengths
    
    
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    decoded_batch = []

    for batch_item in output:
        # argmax over character dimension
        pred = torch.argmax(batch_item, dim=1).cpu().numpy()
        output_indexes = []
        for i,val in enumerate(pred):
            if i>=1:
                if output_indexes[-1]!=val:
                    output_indexes.append(val)
            else:
                output_indexes.append(val)
        output_indexes = [i for i in output_indexes if i != blank_label] # Remove blank label
        output_string = intToStr(output_indexes)
        decoded_batch.append(output_string)

    return decoded_batch


def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    matrix = np.zeros((len(ref) + 1, len(hyp) + 1))
    matrix[0, :] = np.arange(len(hyp) + 1) # Fill the first row of matrix with values ranging from 0 to len(hyp);
    matrix[:, 0] = np.arange(len(ref) + 1) # Fill the first column of matrix with values ranging from 0 to len(ref);
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(matrix[i - 1][j] + 1,matrix[i][j - 1] + 1,matrix[i - 1][j - 1] + 1)

    edit_distance = matrix[len(ref)][len(hyp)]
    return edit_distance    

    
