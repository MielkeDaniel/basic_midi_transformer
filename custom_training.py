import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import tqdm
import random
import argparse
# Add tegridy-tools to Python path
import sys
sys.path.append('./tegridy-tools/tegridy-tools/')
import TMIDIX

from custom_basic_transformer import MusicTransformer, AutoregressiveWrapper


def setup_args():
    parser = argparse.ArgumentParser(description='Music Transformer Training')

    # Model parameters
    parser.add_argument('--seq_len', type=int, default=2048) # 8192
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dim_head', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2) # 4
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=200)

    # Intervals
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--generate_every', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--print_stats_every', type=int, default=20)

    # Paths
    parser.add_argument('--data_path', type=str, default='./content/Training_INTs')
    parser.add_argument('--save_dir', type=str, default='./content/checkpoints')

    return parser.parse_args()


class MusicDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        full_seq = torch.tensor(self.data[index][:self.seq_len + 1]).long()
        return full_seq.cuda()

    def __len__(self):
        return len(self.data)


def setup_training():
    # Configure PyTorch
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


def load_data(path):
    print('Loading processed MIDI data...')
    try:
        train_data = TMIDIX.Tegridy_Any_Pickle_File_Reader(path)
        print('Found existing processed data!')
        return train_data
    except:
        print('No processed data found. Please run MIDI processing first.')
        raise SystemExit(0)


def save_model(model, step, loss, path):
    checkpoint_path = os.path.join(path, f'model_step_{step}_loss_{loss:.4f}.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f'Model saved to {checkpoint_path}')


def plot_training_progress(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses[-100:], label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs[-100:], label='Train')
    plt.plot(val_accs, label='Val')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(args.save_dir, 'training_progress.png'))
    plt.close()


def generate_sample(model, val_dataset, temperature=0.8, seq_len=1024):
    model.eval()
    inp = random.choice(val_dataset)[:64]

    with torch.no_grad():
        sample = model.generate(inp[None, :], seq_len, temperature=temperature)

    generated = []
    time = 0
    dur = 0

    for token in sample[0]:
        if token < 128:  # Time value
            time += token
        elif token < 256:  # Duration
            dur = token - 128
        elif token < 834:  # Pitch
            pitch = token - 256
            generated.append(['note', time, dur, 0, pitch, 90])

    patches = [0] * 16  # Create list of 16 patches, all set to 0 (piano)

    TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
        generated,
        output_signature='Music Transformer',
        output_file_name='generated_sample',
        track_name='Generated Music',
        list_of_MIDI_patches=patches,
        timings_multiplier=32
    )


def train(args):
    setup_training()

    # Load data
    train_data = load_data(args.data_path)
    PAD_IDX = 835
    NUM_TOKENS = PAD_IDX + 1

    # Add to training script after data loading
    print("Sample sequences:")
    for i in range(3):
        print(f"\nSequence {i}:")
        print(train_data[i][:50])  # First 50 tokens

    print("\nToken distribution:")
    all_tokens = [t for seq in train_data for t in seq]
    unique_tokens = set(all_tokens)
    print(f"Unique tokens: {len(unique_tokens)}")
    print(f"Total tokens: {len(all_tokens)}")

    # Initialize model
    model = MusicTransformer(
        num_tokens=NUM_TOKENS,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        dropout=args.dropout
    )

    model = AutoregressiveWrapper(model)
    model.cuda()

    # Initialize optimizer and scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    # Training metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    nsteps = 0

    print('Starting training...')

    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}/{args.num_epochs}')

        # Prepare data
        random.shuffle(train_data)
        split_idx = int(0.9 * len(train_data))
        train_dataset = MusicDataset(train_data[:split_idx], args.seq_len)
        val_dataset = MusicDataset(train_data[split_idx:], args.seq_len)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # Training loop
        model.train()
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            with torch.amp.autocast('cuda'):
                loss, acc = model(data)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if ((i + 1) % args.grad_accum == 0) or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item() * args.grad_accum)
            train_accs.append(acc.item())

            nsteps += 1

            # Print stats
            if i % args.print_stats_every == 0:
                print(f'Step {nsteps} - Loss: {loss.item() * args.grad_accum:.4f}, Acc: {acc.item():.4f}')

            # Validation
            if i % args.validate_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    for val_batch in val_loader:
                        with torch.cuda.amp.autocast():
                            loss, acc = model(val_batch)
                            val_loss += loss.item()
                            val_acc += acc.item()

                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

                    plot_training_progress(train_losses, train_accs, val_losses, val_accs)

                model.train()

            # Generate sample
            if i % args.generate_every == 0:
                generate_sample(model, val_dataset)

            # Save checkpoint
            if i % args.save_every == 0:
                save_model(model, nsteps, train_losses[-1], args.save_dir)


if __name__ == "__main__":
    args = setup_args()
    train(args)