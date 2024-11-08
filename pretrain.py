import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data import Dataloader, batchify, parse_lines
from mygpt import myGPT
from optim import Optim
from tokenizer import BpeTokenizer, Tokenizer


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_type", type=str, choices=["char", "bpe"])
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--ff_embed_dim", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--train_data", type=str, default="./data/train.txt")
    parser.add_argument("--dev_data", type=str, default="./data/val_tiny.txt")
    parser.add_argument("--vocab", type=str, default="./model/vocab.txt")
    parser.add_argument("--min_occur_cnt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--min_len", type=int)
    parser.add_argument("--print_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--save_dir", type=str)

    parser.add_argument("--approx", type=str, default="none")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--MASTER_ADDR", type=str)
    parser.add_argument("--MASTER_PORT", type=str)
    parser.add_argument("--start_rank", type=int)
    parser.add_argument("--backend", type=str)

    args = parser.parse_args()
    return args


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def average_gradients(model):
    normal = True
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        else:
            normal = False
            break
    return normal


def eval_epoch(lm_args, model, tknizer, local_rank, label, batch_acm):
    ds = []
    with open(lm_args.dev_data) as f:
        for line in f:
            ds.append(line.strip())
    ds = parse_lines(ds, lm_args.max_len, lm_args.min_len)

    batch_size = 10
    batches = round(len(ds) / batch_size)
    idx = 0
    avg_nll = 0
    avg_ppl = 0
    avg_acc = 0
    count_bsz = 0
    count_tok = 0
    while idx < len(ds):
        ys_truth, ys_inp, msk = batchify(ds[idx : idx + batch_size], tknizer)
        ys_truth = ys_truth.cuda(local_rank)
        ys_inp = ys_inp.cuda(local_rank)
        msk = msk.cuda(local_rank)
        acc, nll, ppl, toks, bsz = model.ppl(ys_truth, ys_inp, msk)

        avg_nll += nll
        avg_ppl += ppl
        avg_acc += acc
        count_bsz += bsz
        count_tok += toks
        idx += batch_size

    print(
        "validating: label %s, batch_acm %d, acc %.6f, nll %.6f, ppl %.6f"
        % (label, batch_acm, avg_acc / count_tok, avg_nll / count_bsz, avg_ppl / count_bsz),
        flush=True,
    )


def run(args, local_rank):
    torch.manual_seed(1234)
    if args.tokenizer_type == "char":
        tknizer = Tokenizer(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    elif args.tokenizer_type == "bpe":
        tknizer = BpeTokenizer(args.vocab)
    if args.world_size == 1 or dist.get_rank() == 0:
        print("vocab size: %d" % tknizer.size, flush=True)
    model = myGPT(
        local_rank=local_rank,
        vocab=tknizer,
        embed_dim=args.embed_dim,
        ff_embed_dim=args.ff_embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        layers=args.layers,
    )
    if args.start_from is not None:
        cpkt = torch.load(args.start_from, map_location="cpu")
        model.load_state_dict(cpkt["model"])
    model = model.cuda(local_rank)

    if args.world_size > 1:
        torch.manual_seed(1234 + dist.get_rank())
        random.seed(5678 + dist.get_rank())

    optimizer = Optim(
        model.embed_dim,
        args.lr,
        args.warmup_steps,
        torch.optim.AdamW(model.parameters(), lr=1, betas=(0.9, 0.998), eps=1e-9),
    )

    if args.start_from is not None:
        optimizer.load_state_dict(cpkt["optimizer"])

    train_data = Dataloader(
        tknizer=tknizer,
        filename=args.train_data,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_len=args.min_len,
    )
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = 0, 0, 0, 0, 0, 0, 0
    while True:
        if train_data.epoch_id > args.epochs:
            break
        model.train()
        for truth, inp, msk in train_data:
            batch_acm += 1
            print("batch_acm %d" % batch_acm, flush=True)
            truth, inp, msk = truth.cuda(local_rank), inp.cuda(local_rank), msk.cuda(local_rank)
            model.zero_grad()
            res, loss, acc, nll, ppl, ntokens, npairs = model(truth, inp, msk)
            loss_acm += loss.item()
            acc_acm += acc
            nll_acm += nll
            ppl_acm += ppl
            ntokens_acm += ntokens
            npairs_acm += npairs
            nxs += npairs

            loss.backward()
            if args.world_size > 1:
                average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (args.world_size == 1 or dist.get_rank() == 0) and (batch_acm % args.print_every == 0):
                print(
                    f"batch_acm {batch_acm}, loss {loss / args.print_every:.3f}, acc {acc_acm / ntokens_acm:.3f}, "
                    f"nll {nll_acm / nxs:.3f}, ppl {ppl_acm / nxs:.3f}, xacm {npairs_acm}, lr {optimizer.rate():.6f}",
                    flush=True,
                )
                acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = 0, 0, 0, 0, 0, 0, 0
                if (args.world_size == 1 or dist.get_rank() == 0) and (batch_acm % args.save_every == 0):
                    if not os.path.exists(args.save_dir):
                        os.mkdir(args.save_dir)
                    torch.save(
                        {"args": args, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                        f"{args.save_dir}/epoch{train_data.epoch_id}_batch_{batch_acm}",
                    )
                    model.eval()
                    eval_epoch(
                        args,
                        model,
                        tknizer,
                        local_rank,
                        "epoch-" + str(train_data.epoch_id) + "-acm-" + str(batch_acm),
                        batch_acm,
                    )
                    model.train()


def init_processes(args, local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.MASTER_ADDR
    os.environ["MASTER_PORT"] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    fn(args, local_rank)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_config()
    if args.world_size == 1:
        run(args, 0)
        exit(0)
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank, run, args.backend))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
