import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from pyctcdecode.decoder import build_ctcdecoder



# DT2119, Lab 4 End-to-end Speech Recognition


# Variables to be defined --------------------------------------

''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
# (sampling rate for LibriSpeech is always 16 kHz)
train_audio_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(n_mels=80),                # 80 Mel filterbank bins
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),     # masking along freq dimension
    torchaudio.transforms.TimeMasking(time_mask_param=35),          # masking along frame dimension
)


'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = torchaudio.transforms.MelSpectrogram(n_mels=80)



# Functions to be implemented ----------------------------------

### SECTION 3.1 

char2idx = {
    "'": 0,
    " ": 1,
}
for idx, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
    char2idx[char] = idx + 2

def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    # only lowercase strings considered here (LibriSpeech transcripts are uppercase)
    text = text.lower()     
    return [char2idx[char] for char in text]


idx2char = {v: k for k, v in char2idx.items()}

def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    return ''.join(idx2char[label] for label in labels)



### SECTION 3.2

def dataProcessing(data, audio_transform):        # called batch-wise by Torch's DataLoader class
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform (1D tensor), sample_rate, utterance (string), speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x M x T 
            where B=batch_size, C=channels, M=mel_band, T=time_frame.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    mel_spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, _, utterance, _, _, _) in data:
        ### application of audio transform on waveform -> (masked) Mel spectrogram
        # -> Mel spectrum + augmentations (for training data)
        # -> Mel spectrum (for test data)
        mel_spec = audio_transform(waveform)

        # transposing last two dimensions for padding to work (padding function below always pads along dim 0)
        # (nr_mel_bins, nr_frames) -> (nr_frames, nr_mel_bins)
        mel_spec_transposed = mel_spec.squeeze(0).transpose(0, 1)   # squeeze works since mono audio (1 channel)
        mel_spectrograms.append(mel_spec_transposed)

        ### conversion of utterance string into integer labels for CTC
        lbl = torch.tensor(strToInt(utterance))
        labels.append(lbl)

        ### keeping track of input and label lengths for each utterance in batch
        # !!! division by 2 needed for mel_specs due to halving of time dimension (nr_frames) by first CNN layer
        # which uses stride 2 !!!
        # -> since CTC loss operates on output side, so lengths it gets must reflect that downsampling
        input_lengths.append(mel_spec_transposed.shape[0] // 2)
        label_lengths.append(lbl.shape[0])

    ### padding along time dimension to length of longest utterance (max nr of frames)
    # -> (frame_i, nr_mel_bins) -> (batch_size, frame_max, nr_mel_bins)
    mel_spectrograms_padded = pad_sequence(sequences=mel_spectrograms, batch_first=True, padding_value=0)

    # reshaping to expected input format (batch_size, nr_channels, nr_mel_bins, nr_frames)
    mel_spectrograms_input = mel_spectrograms_padded.unsqueeze(1).transpose(2, 3)

    # same for labels (to length of longest character string)
    labels_input = pad_sequence(sequences=labels, batch_first=True, padding_value=0)   # pad with blank symbol ?

    return mel_spectrograms_input, labels_input, input_lengths, label_lengths


# example = torch.load('lab4_example.pt', weights_only=False)

# spec, lab, in_len, lab_len = dataProcessing(example['data'], test_audio_transform)

# print(spec.shape,    'vs', example['spectrograms'].shape)
# print(lab.shape,     'vs', example['labels'].shape)
# print(in_len,        'vs', example['input_lengths'])
# print(lab_len,       'vs', example['label_lengths'])

# print('spec close:',  torch.allclose(spec, example['spectrograms']))
# print('labels equal:', torch.equal(lab, example['labels']))

    
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    # picking maximum likelihood character per frame
    max_chars = torch.argmax(output, dim=2)   # shape (batch_size, nr_frames//2) due to batching

    # decoded outputs (converting class indices to string characters)
    collapsed_unblanked_chars = []
    for pred_labels_utterance in max_chars:
        pred_labels_utterance = pred_labels_utterance.tolist()

        collapsed_unblanked_labels = []
        for i, label_idx in enumerate(pred_labels_utterance):
            # only keep unrepeated labels (equiv to collapse) and ignore blanks
            if (label_idx != blank_label) and (i == 0 or label_idx != pred_labels_utterance[i - 1]):
                collapsed_unblanked_labels.append(label_idx)

        collapsed_unblanked_chars.append(intToStr(collapsed_unblanked_labels))

    return collapsed_unblanked_chars    


def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    ref_len, hyp_len = len(ref), len(hyp)

    # initialisation of distance matrix 
    distance_matrix = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    # first row and column can be initialised 
    for i in range(ref_len + 1):
        distance_matrix[i][0] = i
    
    for j in range(hyp_len + 1):
        distance_matrix[0][j] = j

    # minimum edit distance algorithm
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            else:
                distance_matrix[i][j] = 1 + min(
                    distance_matrix[i - 1][j],          # deletion from ref
                    distance_matrix[i][j - 1],          # insertion into ref
                    distance_matrix[i - 1][j - 1],      # substitution
                )
    return distance_matrix[ref_len][hyp_len]


# pyctcdecode convention: blank corresponds to empty string ""
labels_list = [idx2char[i] for i in range(28)] + [""]

beam_search_decoder = build_ctcdecoder(
    labels_list,
    kenlm_model_path='./wiki-interpolate.3gram.arpa',
    alpha=0.5,
    beta=1.0,
)

def languageDecoder(output, decoder=beam_search_decoder):
    '''
    Decodes a batch using beam search with a language model.

    output: (B, T, C) tensor of log-probabilities
    returns: list of decoded strings
    '''
    decoded = []
    # pyctcdecode can only take a single utterance's logits at a time, so looping needed
    for i in range(output.shape[0]):
        logits = output[i].cpu().detach().numpy()   # (T, C) numpy array, pyctcdecode works with NumPy
        text = decoder.decode(logits)               # now uses model and LM probs weighting for inference 
        decoded.append(text)
    return decoded
