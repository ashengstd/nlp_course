import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time

from mygpt import myGPT
from tokenizer import Tokenizer
from data import  DataLoader, s2t

mstime = lambda: int(round(time.time() * 1000))    

def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Tokenizer(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = myGPT(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

if __name__ == "__main__":
    device = 7
    print("loading...")
    m_path = "./ckpt/epoch0_batch_199"
    v_path = "./model/vocab.txt"
    lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")
    
    max_len  = 50
    qs = ["介绍下南京航空航天大学", "Please introduce Nanjing University of Aeronautics and Astronautics"]
    print(qs)
    i = 0
    for q in qs:
        start = mstime()
        i += 1
        s = [[w for w in q]]

        r1 = greedy(lm_model, lm_vocab, device, s, max_len)

        r2 = beam_search(lm_model, lm_vocab, device, s, max_len)

        r3 = top_k_inc(lm_model, lm_vocab, device, s, 5, max_len)

        r4 = top_k_inc(lm_model, lm_vocab, device, s, 10, max_len)

        r5 = top_k_inc(lm_model, lm_vocab, device, s, 20, max_len)

        r6 = top_k_inc(lm_model, lm_vocab, device, s, 50, max_len)

        r7 = top_k_inc(lm_model, lm_vocab, device, s, 500, max_len)

        r8 = top_p_inc(lm_model, lm_vocab, device, s, 20, 0.95, max_len)
    
        print(i)
        print("q: ", q)
        print("greedy: ", r1)
        print("bm5: ", q+r2)
        print("tk5: ", r3)
        print("tk10: ", r4)
        print("tk20: ", r5)
        print("tk50: ", r6)
        print("tk500: ", r7)
        print("tp0.95: ", r8)
        print(mstime()-start)
    
