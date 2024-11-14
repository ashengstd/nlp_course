import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from base_model.mygpt import MyGPT
from base_model.optim import Optim
from base_model.tokenizer import BpeTokenizer, Tokenizer
from utils.data import DataLoader, batchify, parse_lines


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_type", type=str, default="char", choices=["char", "bpe"])
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--ff_embed_dim", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, default="./data/train.txt")
    parser.add_argument("--dev_data", type=str, default="./data/val_tiny.txt")
    parser.add_argument("--vocab", type=str, default="./model/vocab.txt")
    parser.add_argument("--min_occur_cnt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="ckpt")
    parser.add_argument("--approx", type=str, default="none")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--MASTER_ADDR", type=str, default="localhost")
    parser.add_argument("--MASTER_PORT", type=str, default="25555")
    parser.add_argument("--start_rank", type=int, default=0)
    parser.add_argument("--backend", type=str, default="nccl")
    return parser.parse_args()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def average_gradients(model):
    normal = True
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
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
            line = line.strip()
            if line:
                ds.append(line)

    ds = parse_lines(ds, lm_args.max_len, lm_args.min_len)
    batch_size = 10
    idx = 0
    avg_nll, avg_ppl, avg_acc = 0.0, 0.0, 0.0
    count_bsz, count_tok = 0.0, 0.0

    while idx < len(ds):
        ys_truth, ys_inp, msk = batchify(ds[idx : idx + batch_size], tknizer)
        ys_truth, ys_inp, msk = ys_truth.cuda(local_rank), ys_inp.cuda(local_rank), msk.cuda(local_rank)

        acc, nll, ppl, toks, bsz = model.ppl(ys_truth, ys_inp, msk)
        avg_acc += acc
        avg_nll += nll
        avg_ppl += ppl
        count_bsz += bsz
        count_tok += toks
        idx += batch_size

    print(
        f"validating: label {label}, batch_acm {batch_acm}, "
        f"acc {avg_acc / count_tok:.6f}, nll {avg_nll / count_bsz:.6f}, "
        f"ppl {avg_ppl / count_bsz:.6f}",
        flush=True,
    )


def save_model(args, model, optimizer, train_data, batch_acm):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # for filename in os.listdir(args.save_dir):
    #     file_path = os.path.join(args.save_dir, filename)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

    torch.save(
        {"args": args, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        f"{args.save_dir}/epoch{train_data.epoch_id}_batch_{batch_acm}",
    )


def run(args, local_rank):
    torch.manual_seed(1234)
    tknizer = (
        Tokenizer(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
        if args.tokenizer_type == "char"
        else BpeTokenizer(args.vocab, specials=[])
    )
    if args.world_size == 1 or dist.get_rank() == 0:
        print(f"vocab.size = {tknizer.size}", flush=True)

    model = MyGPT(local_rank, tknizer, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.layers)
    if args.start_from is not None:
        ckpt = torch.load(args.start_from, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    model = model.cuda(local_rank)

    if args.world_size > 1:
        torch.manual_seed(1234 + dist.get_rank())
        random.seed(5678 + dist.get_rank())

    optimizer = Optim(
        model.embed_dim,
        args.lr,
        args.warmup_steps,
        torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9),
    )

    if args.start_from is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    train_data = DataLoader(tknizer, args.train_data, args.batch_size, args.max_len, args.min_len)
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    while train_data.epoch_id <= args.epoch:
        model.train()
        for truth, inp, msk in train_data:
            batch_acm += 1
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

            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.print_every == 0:
                print(
                    (
                        f"batch_acm {batch_acm}, loss {loss_acm / args.print_every:.3f}, "
                        f"acc {acc_acm / ntokens_acm:.3f}, nll {nll_acm / nxs:.3f}, "
                        f"ppl {ppl_acm / nxs:.3f}, x_acm {npairs_acm}, lr {optimizer._rate:.6f}"
                    ),
                    flush=True,
                )
                acc_acm, nll_acm, ppl_acm, ntokens_acm, loss_acm, nxs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.save_every == 0:
                save_model(args, model, optimizer, train_data, batch_acm)
                model.eval()
                eval_epoch(args, model, tknizer, local_rank, f"epoch-{train_data.epoch_id}-acm-{batch_acm}", batch_acm)
                model.train()


def init_processes(args, local_rank, fn, backend="nccl"):
    os.environ["MASTER_ADDR"] = args.MASTER_ADDR
    os.environ["MASTER_PORT"] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    fn(args, local_rank)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_config()
    if args.world_size == 1:
        run(args, 0)
    else:
        processes = []
        for rank in range(args.gpus):
            p = mp.Process(target=init_processes, args=(args, rank, run, args.backend))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
