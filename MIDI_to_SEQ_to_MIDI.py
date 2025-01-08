import os
import sys
sys.path.append('./tegridy-tools/tegridy-tools/')
import TMIDIX

def process_midi_to_sequence(midi_file):
    raw_score = TMIDIX.midi2single_track_ms_score(midi_file)
    raw_notes = [event for event in raw_score[1] if event[0] == 'note']
    
    sequence = [834]  # Start token
    current_time = 0
    
    TIMING_FACTOR = 32  # Increased from 8 for slower playback
    
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

def sequence_to_midi(sequence, output_name='test_output'):
    if sequence[0] == 834:
        sequence = sequence[1:]
    
    generated = []
    time = 0
    TIMING_FACTOR = 32  # Match the processing factor
    
    i = 0
    while i < len(sequence):
        if sequence[i] < 128:
            time += sequence[i] * TIMING_FACTOR
            i += 1
        elif i + 1 < len(sequence):
            duration = (sequence[i] - 128) * TIMING_FACTOR
            pitch = sequence[i + 1] - 256
            generated.append(['note', time, duration, 0, pitch, 90])
            i += 2
        else:
            i += 1
    
    patches = [0] * 16
    TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
        generated,
        output_signature='Test',
        output_file_name=output_name,
        track_name='Generated Music',
        list_of_MIDI_patches=patches
    )
    return generated

if __name__ == "__main__":
    dataset_addr = "./content/dataset"
    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        if filenames:
            midi_file = os.path.join(dirpath, filenames[0])
            print(f"Processing {midi_file}")
            seq = process_midi_to_sequence(midi_file)
            generated = sequence_to_midi(seq, 'test_output')
            print("First 5 MIDI events:", generated[:5])
            break