import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
from torch.nn.utils import clip_grad_norm_
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model, print_trainable_params
from src.model import load_model
from src.dataset import load_dataset
from src.utils.collate import collate_funcs


def main(args):
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    collate_fn = collate_funcs[args.dataset](dataset.graph)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model[args.model_name](
        in_channels=dataset.num_features,
        hidden_channels=args.gnn_hidden_dim,
        out_channels=dataset.num_classes,
        num_layers=args.gnn_num_layers,
        num_heads=args.gnn_num_heads,
        dropout=args.gnn_dropout,
    ).to(device)

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd},],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = print_trainable_params(model)
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss, best_val_acc = float('inf'), -float('inf')
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            pred, _ = model(x, edge_index)
            loss = loss_func(pred.cpu()[batch['mapping']], batch['y'][batch['mapping']])
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr})
                wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.

            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        val_loss = 0.
        correct = 0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                x = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                mapping = batch['mapping']
                pred, _ = model(x, edge_index)
                loss = loss_func(pred.cpu()[mapping], batch['y'][mapping])
                correct += (pred.cpu().argmax(dim=-1)[mapping] == batch['y'][mapping]).sum().item()
                val_loss += loss.item()
            val_loss = val_loss/len(val_loader)
            val_acc = correct / len(val_dataset)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss} Val Acc: {val_acc}")
            wandb.log({'Val Loss': val_loss, 'Val Acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5. Evaluating
    model = _reload_best_model(model, args)
    model.eval()

    progress_bar_test = tqdm(range(len(test_loader)))
    correct = 0
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            pred, _ = model(x, edge_index)
            correct += (pred.cpu().argmax(dim=-1) == batch['y'])[batch['mapping']].sum().item()
        progress_bar_test.update(1)
    acc = correct / len(test_dataset)

    print(f'Test Acc: {acc}')
    wandb.log({'Test Acc': acc})


if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
