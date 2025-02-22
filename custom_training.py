import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import TMIDIX
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import tqdm
import random
from dataset import MIDI_Dataset, detokenize
from custom_basic_transformer import MusicTransformer, AutoregressiveWrapper


class Args:
    def __init__(self):
        self.seq_len = 100
        self.dim = 512
        self.depth = 6
        self.heads = 8
        self.dim_head = 64
        self.dropout = 0.1
        self.batch_size = 2
        self.grad_accum = 2
        self.lr = 3e-4
        self.num_epochs = 60
        self.validate_every = 100
        self.generate_every = 200
        self.save_every = 500
        self.print_stats_every = 20
        self.data_path = './content/dataset'
        self.save_dir = './content/checkpoints'



def setup_training():
    # Configure PyTorch
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists('./content/samples'):
        os.makedirs('./content/samples')


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


def generate_sample(model, val_dataset, step, temperature=0.8, seq_len=1024):
    model.eval()
    inp = random.choice(val_dataset)[:50]

    with torch.no_grad():
        sample = model.generate(inp[None, :], seq_len, temperature=temperature)

    detokenize(sample[0], f'./content/samples/sample_step_{step}.mid')


def train(args):
    # Load data
    print('Creating MIDI dataset...')
    dataset = MIDI_Dataset(path=args.data_path, max_seq_len=args.seq_len)
    PAD_IDX = dataset.pad_idx
    NUM_TOKENS = PAD_IDX + 1

    # Print dataset statistics 
    print(f"Total sequences: {len(dataset)}")

    # Initialize model with correct token range
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

    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
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

       # Prepare data splits
        split_idx = int(0.9 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])

        train_loader = DataLoader(train_dataset, 
                         batch_size=args.batch_size, 
                         shuffle=True,
                         collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

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
                generate_sample(model, val_dataset, nsteps)

            # Save checkpoint
            if i % args.save_every == 0:
                save_model(model, nsteps, train_losses[-1], args.save_dir)


if __name__ == "__main__":
    args = Args()
    setup_training()
    train(args)