import random
import os
import torch
from torch.utils.data import Dataset
import TMIDIX

class MIDI_Dataset(Dataset):
    def __init__(self, path='./content/909_dataset', max_seq_len=128, pad_idx=835, rand_start=True, shuffle=True, aug=False):
        self.rand_start = rand_start
        self.midi_list = get_midi_list(path)
        if shuffle:
            random.shuffle(self.midi_list)
        self.max_seq_len = max_seq_len 
        self.aug = aug
        self.pad_idx = pad_idx
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.TIMING_FACTOR = 32
        self.sequence_lengths = []
        self.counter = 0
    
    def __len__(self):
        return len(self.midi_list)
    
    def get_midi_list(self):
        return self.midi_list
    
    def transpose_augmentation(self, sequence):
        # Random transposition between -2 and +2 semitones
        transpose_amount = random.randint(-2, 2)
        
        if transpose_amount == 0:
            return sequence
            
        # Convert to list if tensor
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        
        transposed = []
        for token in sequence:
            if token >= 256 and token < 834:  # Only transpose pitch tokens
                # Ensure pitch stays in valid MIDI range (0-127)
                new_pitch = ((token - 256) + transpose_amount)
                if 0 <= new_pitch <= 127:
                    transposed.append(new_pitch + 256)
                else:
                    transposed.append(token)  # Keep original if out of range
            else:
                transposed.append(token)
                
        return torch.tensor(transposed, device=self.device).long()

    def __getitem__(self, idx):
        try:
            raw_score = TMIDIX.midi2single_track_ms_score(self.midi_list[idx]) 
            raw_notes = [event for event in raw_score[1] if event[0] == 'note']
            
            # minimum notes threshold, otherwise do not return none but a random sample
            if len(raw_notes) < 40:  # Minimum notes threshold
                self.counter += 1 
                return self.__getitem__(random.randint(0, len(self) - 1))

            sequence = [834]  # Start token
            current_time = 0
            for note in raw_notes:
                time_diff = note[1] - current_time
                if time_diff > 0:
                    sequence.append(min(time_diff // self.TIMING_FACTOR, 127))

                duration = min(note[2] // self.TIMING_FACTOR, 127)
                pitch = note[4]

                sequence.append(duration + 128)
                sequence.append(pitch + 256)
                current_time = note[1]

            # even if rand_start is true, only do it in 85% of the cases
            if self.rand_start and len(sequence) > self.max_seq_len and random.random() < 0.85:
                start = random.randint(0, len(sequence) - self.max_seq_len - 1)
                sequence = sequence[start:start+self.max_seq_len+1]
            else:
                sequence = sequence[:self.max_seq_len+1] # +1 for input/target pair creation

            if self.aug:
                sequence = self.transpose_augmentation(sequence)
            return torch.tensor(sequence, device=self.device).long()
        except Exception as e:
            print(f'Error processing file {self.midi_list[idx]}: {str(e)}')
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def collate_fn(self, batch):
        # Convert tensors to same length and stack
        padded_batch = torch.nn.utils.rnn.pad_sequence(batch, 
                                                    batch_first=True,
                                                    padding_value=self.pad_idx)
        return padded_batch.to(self.device)


def get_midi_list(path):
    filez = list()
    for (dirpath, _, filenames) in os.walk(path):
        filez += [os.path.join(dirpath, file) for file in filenames]
    return filez


def detokenize(sample, output_file="output"): 
    generated = []
    time = 0
    dur = 0

    for token in sample:
        token = token.item()
        if token < 128:  # Time value
            time += token
        elif token < 256:  # Duration
            dur = token - 128
        elif token < 834:  # Pitch
            pitch = token - 256
            generated.append(['note', time, dur, 0, pitch, 90])

    # Write to MIDI file
    patches = [0] * 16  # Piano patches
    TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
        generated,
        output_signature='Music Transformer',
        output_file_name=output_file,
        track_name='Generated Music',
        list_of_MIDI_patches=patches,
        timings_multiplier=32
    )

# print amount of tokens in the dataset 
def print_tokens(dataset):
    tokens = 0
    for i in range(len(dataset)):
        tokens += len(dataset[i])
        # print every 100 files
        if i % 100 == 0:
            print(f"Processed {i} files with {tokens} tokens")
    print(f"Total tokens: {tokens}")


#12.000.000 tokens in 909 dataset --> 600.000 parameters in model (12M / 20)
# 100.000 tokens in dataset --> 5.000 parameters in model (100 / 20)
