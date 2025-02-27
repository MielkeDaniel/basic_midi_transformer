import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import matplotlib.pyplot as plt
import tqdm
import random
from dataset import MIDI_Dataset, detokenize
from custom_basic_transformer import MusicTransformer, AutoregressiveWrapper, printModelParameters


class Args:
    def __init__(self):
        #dim=128, depth=3, heads=4, dim_head=32
        self.seq_len = 512
        self.dim = 512
        self.depth = 6
        self.heads = 8
        self.dim_head = 32
        self.dropout = 0.1
        self.batch_size = 32
        self.grad_accum = 2
        self.lr = 8e-5
        self.num_epochs = 1000
        self.validate_every = 600
        self.generate_every = 1000
        self.weight_decay = 0.01
        self.save_every = 10000
        self.print_stats_every = 100
        self.checkpoint = None #'./content/checkpoints/checkpoint.pt'
        self.data_path = './content/909_dataset'
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
    """
    Plot the training and validation losses and accuracies.
    Training data is plotted against training steps.
    Validation data is plotted against validation rounds.
    """
    current_step = len(train_losses)
    
    plt.figure(figsize=(15, 6))
    
    # Plot losses
    ax1 = plt.subplot(1, 2, 1)
    
    # For training, use step indices
    train_steps = list(range(1, current_step + 1))
    
    # For validation, use validation round numbers
    val_indices = list(range(1, len(val_losses) + 1))
    
    # Plot training losses
    if current_step > 1000:
        # Sample for large datasets
        sparse_idx = list(range(0, current_step, 10))
        ax1.plot([train_steps[i] for i in sparse_idx], 
                 [train_losses[i] for i in sparse_idx], 
                 'b-', alpha=0.3, label='Train (sampled)')
        # Always show recent detail
        ax1.plot(train_steps[-100:], train_losses[-100:], 'b-', linewidth=1.5, label='Train (recent)')
    else:
        ax1.plot(train_steps, train_losses, 'b-', alpha=0.6, label='Train')
    
    # Create second axis for validation
    ax2 = ax1.twiny()
    line_val, = ax2.plot(val_indices, val_losses, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Validation Round')
    ax2.set_xlim(0.5, len(val_losses) + 0.5)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss (Step {current_step})')
    
    # Add validation line to ax1's legend
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(line_val)
    labels.append('Validation')
    ax1.legend(handles, labels, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax3 = plt.subplot(1, 2, 2)
    
    # Plot training accuracies
    if current_step > 1000:
        ax3.plot([train_steps[i] for i in sparse_idx], 
                 [train_accs[i] for i in sparse_idx], 
                 'b-', alpha=0.3, label='Train (sampled)')
        ax3.plot(train_steps[-100:], train_accs[-100:], 'b-', linewidth=1.5, label='Train (recent)')
    else:
        ax3.plot(train_steps, train_accs, 'b-', alpha=0.6, label='Train')
    
    # Create second axis for validation
    ax4 = ax3.twiny()
    line_val_acc, = ax4.plot(val_indices, val_accs, 'r-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Validation Round')
    ax4.set_xlim(0.5, len(val_losses) + 0.5)
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Accuracy')
    ax3.set_title(f'Accuracy (Step {current_step})')
    
    # Add validation line to ax3's legend
    handles, labels = ax3.get_legend_handles_labels()
    handles.append(line_val_acc)
    labels.append('Validation')
    ax3.legend(handles, labels, loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save both a step-specific version and the latest version
    # plt.savefig(os.path.join(args.save_dir, f'training_progress_step_{current_step}.png'), dpi=150)
    plt.savefig(os.path.join(args.save_dir, 'latest_training_progress.png'), dpi=150)
    plt.close()


def generate_sample(model, val_dataset, step, temperature=0.9, seq_len=256):
    model.eval()
    inp = random.choice(val_dataset)[:50]

    with torch.no_grad():
        sample = model.generate(inp[None, :], seq_len, temperature=temperature)

    detokenize(sample[0], f'./content/samples/sample_step_{step}.mid')


def train(args):
    # Load data
    print('Creating MIDI dataset...')
    dataset = MIDI_Dataset(path=args.data_path, max_seq_len=args.seq_len, aug=False)
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

    # Load checkpoint if available
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.endswith('rope.cache')}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f'Model loaded from {args.checkpoint}')

    model.cuda()
    printModelParameters(model)

    # Use AdamW optimizer
    optimizer = optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'norm'])],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'norm'])],
             'weight_decay': 0.0}
        ],
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )
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