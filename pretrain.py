import os

import torch.distributed as dist
import torch.multiprocessing as mp


def parse_config():
    pass


def run(args, local_rank):
    pass


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
