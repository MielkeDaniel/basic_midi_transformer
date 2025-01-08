import os
import tqdm
import random
from joblib import Parallel, delayed, parallel_config
import sys
sys.path.append('./tegridy-tools/tegridy-tools/')
import TMIDIX

TIMING_FACTOR = 32
SEQ_LEN = 256
PAD_IDX = 835

def process_midi(midi_file):
    try:
        raw_score = TMIDIX.midi2single_track_ms_score(midi_file)
        raw_notes = [event for event in raw_score[1] if event[0] == 'note']

        if len(raw_notes) < 64:  # Minimum notes threshold
            return None

        sequence = [834]  # Start token
        current_time = 0

        for note in raw_notes:
            time_diff = note[1] - current_time
            if time_diff > 0:
                sequence.append(min(time_diff // TIMING_FACTOR, 127))

            duration = min(note[2] // TIMING_FACTOR, 127)
            pitch = note[4]

            sequence.append(duration + 128)
            sequence.append(pitch + 256)
            current_time = note[1]

        return sequence
    except Exception as e:
        print(f'Error processing {midi_file}: {e}')
        return None

def prepare_training_data(sequences):
    train_data = []

    for seq in sequences:
        if seq is None or len(seq) < 4:  # Need at least one note (time + duration + pitch)
            continue

        # Handle sequences shorter than SEQ_LEN
        if len(seq) <= SEQ_LEN + 1:
            padded = seq + [PAD_IDX] * (SEQ_LEN + 1 - len(seq))
            train_data.append(padded)
        else:
            # Create overlapping sequences for longer pieces
            for i in range(0, len(seq) - SEQ_LEN, SEQ_LEN // 2):
                train_data.append(seq[i:i + SEQ_LEN + 1])

    return train_data

if __name__ == "__main__":
    # Setup directories
    os.makedirs('./content/dataset', exist_ok=True)
    os.makedirs('./content/INTS', exist_ok=True)

    # Get MIDI files
    dataset_addr = "./content/dataset"
    filez = []
    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        filez += [os.path.join(dirpath, file) for file in filenames]
    random.shuffle(filez)

    # Process MIDI files in parallel
    print('Processing MIDI files...')
    with parallel_config(backend='threading', n_jobs=8, verbose=0):
        sequences = Parallel()(delayed(process_midi)(f) for f in tqdm.tqdm(filez))

    sequences = [s for s in sequences if s is not None]

    # Prepare training data
    print('Preparing training data...')
    train_data = prepare_training_data(sequences)
    random.shuffle(train_data)

    print(f'Generated {len(train_data)} training sequences')

    # Save processed data
    TMIDIX.Tegridy_Any_Pickle_File_Writer(sequences, './content/Processed_MIDIs')
    TMIDIX.Tegridy_Any_Pickle_File_Writer(train_data, './content/Training_INTs')