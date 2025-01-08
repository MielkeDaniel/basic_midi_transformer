import sys
import os
sys.path.append('./tegridy-tools/tegridy-tools/')
import TMIDIX

def debug_midi_processing(midi_file):
    print("1. Converting MIDI to score...")
    raw_score = TMIDIX.midi2single_track_ms_score(midi_file)
    print("Raw score first 5 events:", raw_score[1][:5] if len(raw_score) > 1 else "Empty score")
    
    print("\n2. Processing score...")
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    print("Enhanced score first 5 notes:", escore_notes[:5] if escore_notes else "Empty enhanced score")
    
    print("\n3. Augmenting notes...")
    aug_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=8)
    print("Augmented first 5 notes:", aug_notes[:5] if aug_notes else "Empty augmented notes")
    
    print("\n4. Creating delta score...")
    delta_score = [d[1:] for d in TMIDIX.delta_score_notes(aug_notes, even_timings=False, compress_timings=False)]
    print("Delta score first 5 notes:", delta_score[:5] if delta_score else "Empty delta score")
    
    return delta_score

if __name__ == "__main__":
    # Get first MIDI file path
    dataset_addr = "./content/dataset"
    midi_files = []
    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        midi_files += [os.path.join(dirpath, file) for file in filenames]
        break  # Only process root directory
    
    if midi_files:
        print(f"Processing {midi_files[0]}")
        processed = debug_midi_processing(midi_files[0])
        
        # Convert processed data back to MIDI for verification
        generated = []
        time = 0
        
        for note in processed:
            if len(note) >= 3:  # Ensure we have time, duration, pitch
                time += note[0] if note[0] != 0 else 0
                generated.append(['note', time, note[1], 0, note[2], 90])
        
        if generated:
            patches = [0] * 16
            TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
                generated,
                output_signature='Debug',
                output_file_name='debug_output',
                track_name='Debug Track',
                list_of_MIDI_patches=patches,
                timings_multiplier=8
            )
            print("\nGenerated MIDI events:", generated[:5])
        else:
            print("\nNo events generated!")
    else:
        print("No MIDI files found in dataset directory")