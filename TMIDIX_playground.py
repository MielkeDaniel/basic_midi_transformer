import sys
sys.path.append('./tegridy-tools/tegridy-tools/')
import TMIDIX

raw_score = TMIDIX.midi2single_track_ms_score('./content/dataset/8_transpose_0.mid')

print(raw_score)

