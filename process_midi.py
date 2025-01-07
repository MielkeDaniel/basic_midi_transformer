import os
import tqdm
import random
from joblib import Parallel, delayed, parallel_config

# Add tegridy-tools to Python path
import sys
sys.path.append('./tegridy-tools/tegridy-tools/')
import TMIDIX

if not os.path.exists('./content/dataset'):
    os.makedirs('./content/dataset')
if not os.path.exists('./content/INTS'):
    os.makedirs('./content/INTS')

dataset_addr = "./content/dataset"
filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
random.shuffle(filez)


def TMIDIX_MIDI_Processor(midi_file):
    try:
        raw_score = TMIDIX.midi2single_track_ms_score(midi_file)
        escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=16)

        # Instead of taking just first note:
        delta_score = [d[1:] for d in TMIDIX.delta_score_notes(escore_notes,
                                                          even_timings=True,
                                                          compress_timings=True)]
        return delta_score

    except Exception as e:
        print('Error:', e)
        return None


melody_chords_f = []
for i in tqdm.tqdm(range(0, len(filez), 16)):
    with parallel_config(backend='threading', n_jobs=16, verbose=0):
        output = Parallel()(delayed(TMIDIX_MIDI_Processor)(f) for f in filez[i:i + 16])
        for o in output:
            if o is not None:
                melody_chords_f.append(o)

SEQ_LEN = 8192
PAD_IDX = 835

train_data = []
for melody in tqdm.tqdm(melody_chords_f):
    dat = [834]  # Start token

    for note in melody:
        if note[0] != 0:  # Time value
            dat.append(note[0])
        dat.append(note[1] + 128)  # Duration
        dat.append(note[2] + 256)  # Pitch

    dat = dat[:SEQ_LEN + 1]
    dat += [PAD_IDX] * (SEQ_LEN + 1 - len(dat))
    train_data.append(dat)

random.shuffle(train_data)

TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, './content/Processed_MIDIs')
TMIDIX.Tegridy_Any_Pickle_File_Writer(train_data, './content/Training_INTs')