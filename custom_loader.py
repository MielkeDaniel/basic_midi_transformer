import torch
import sys
sys.path.append('./tegridy-tools/tegridy-tools/')
import TMIDIX
from custom_basic_transformer import MusicTransformer, AutoregressiveWrapper

# Model parameters (matching training)
PAD_IDX = 835
NUM_TOKENS = PAD_IDX + 1
dim = 512
depth = 6
heads = 8
dim_head = 64
dropout = 0.1

# Initialize model with same architecture
model = MusicTransformer(
    num_tokens=NUM_TOKENS,
    dim=dim,
    depth=depth,
    heads=heads,
    dim_head=dim_head,
    dropout=dropout
)

model = AutoregressiveWrapper(model)

# Load trained weights
checkpoint = torch.load('./models/v1.0.pt', weights_only=True)
state_dict = checkpoint['model_state_dict']

# Remove cache entries
state_dict = {k: v for k, v in state_dict.items() if 'rope.cache' not in k}

model.load_state_dict(state_dict)
model.cuda()
# After model.cuda()
torch.cuda.empty_cache()  # Clear CUDA memory cache
model.eval()

# Generate sequence
with torch.no_grad():
    # Start with token 834 (Start token from training)
    start_tokens = torch.tensor([[834]], dtype=torch.long).cuda()

    # Generate 1024 tokens with slight temperature
    sample = model.generate(start_tokens, 1024, temperature=0.8)

print(sample[:, :100])
# Convert tokens to MIDI events
generated = []
time = 0
dur = 0

for token in sample[0]:
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
    output_file_name='./content/generated_sample',
    track_name='Generated Music',
    list_of_MIDI_patches=patches,
    timings_multiplier=32
)