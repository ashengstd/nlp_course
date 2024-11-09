import argparse
import os

import torch

from data import DataLoader, batchify, parse_lines
from mygpt import myGPT
from optim import Optim
from tokenizer import Tokenizer


def parse_config():
    parser = argparse.ArgumentParser()
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

    return parser.parse_args()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def eval_epoch(lm_args, model, tknizer, label, batch_acm):
    ds = []
    with open(lm_args.dev_data) as f:
        for line in f:
            line = line.strip()
            if line:
                ds.append(line)

    ds = parse_lines(ds, lm_args.max_len, lm_args.min_len)
    batch_size = 10
    idx = 0
    avg_nll = 0.0
    avg_ppl = 0.0
    avg_acc = 0.0
    count_bsz = 0.0
    count_tok = 0.0

    while idx < len(ds):
        ys_truth, ys_inp, msk = batchify(ds[idx : idx + batch_size], tknizer)
        ys_truth, ys_inp, msk = ys_truth.cuda(), ys_inp.cuda(), msk.cuda()

        acc, nll, ppl, toks, bsz = model.ppl(ys_truth, ys_inp, msk)
        avg_acc += acc
        avg_nll += nll
        avg_ppl += ppl
        count_bsz += bsz
        count_tok += toks
        idx += batch_size

    print(
        "validating: label %s, batch_acm %d, acc %.6f, nll %.6f, ppl %.6f"
        % (label, batch_acm, avg_acc / count_tok, avg_nll / count_bsz, avg_ppl / count_bsz),
        flush=True,
    )


def run(args):
    torch.manual_seed(1234)
    tknizer = Tokenizer(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    print("vocab.size = %d" % tknizer.size, flush=True)

    model = myGPT(0, tknizer, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.layers)
    if args.start_from is not None:
        ckpt = torch.load(args.start_from, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    model = model.cuda()

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
            truth, inp, msk = truth.cuda(), inp.cuda(), msk.cuda()

            model.zero_grad()
            res, loss, acc, nll, ppl, ntokens, npairs = model(truth, inp, msk)
            acc_acm += acc
            nll_acm += nll
            ppl_acm += ppl
            ntokens_acm += ntokens
            npairs_acm += npairs
            nxs += npairs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_acm % args.print_every == 0:
                print(
                    "batch_acm %d, loss %.3f, acc %.3f, nll%.3f, ppl %.3f, x_acm %d, lr %.6f"
                    % (
                        batch_acm,
                        loss_acm / args.print_every,
                        acc_acm / ntokens_acm,
                        nll_acm / nxs,
                        ppl_acm / nxs,
                        npairs_acm,
                        optimizer._rate,
                    ),
                    flush=True,
                )
                acc_acm, nll_acm, ppl_acm, ntokens_acm, loss_acm, nxs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if batch_acm % args.save_every == 0:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save(
                    {"args": args, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    "%s/epoch%d_batch_%d" % (args.save_dir, train_data.epoch_id, batch_acm),
                )

                model.eval()
                eval_epoch(
                    args, model, tknizer, "epoch-" + str(train_data.epoch_id) + "-acm-" + str(batch_acm), batch_acm
                )
                model.train()


if __name__ == "__main__":
    args = parse_config()
    run(args)
