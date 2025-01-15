# based of the paper "Titans: Learning to Memorize at Test Time"
# https://arxiv.org/pdf/2501.00663v1

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Optional

class DeepMemoryModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        sizes = [in_dim] + [hidden_dim]*(num_layers-1) + [out_dim]
        for i in range(num_layers):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1], bias=True))
            if i < num_layers-1:
                self.layers.append(nn.SiLU())
        self.momentum_buffers = {}
        self.alpha = nn.Parameter(torch.tensor(0.0001))
        self.theta = nn.Parameter(torch.tensor(0.0001))
        self.eta = nn.Parameter(torch.tensor(0.9))

    def forward_no_update(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def forward_with_update(self, k, v):
        pred = self.forward_no_update(k)
        diff = (pred - v)
        loss = 0.5*(diff**2).sum()
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=False)
        idx = 0
        with torch.no_grad():
            for param in self.parameters():
                g = grads[idx]
                idx += 1
                if g is None:
                    continue
                if not hasattr(param, "surprise"):
                    param.surprise = torch.zeros_like(param.data)
                param.surprise = (self.eta*param.surprise) - (self.theta*g)
                param.data = (1 - self.alpha)*param.data + param.surprise

    def forward(self, x):
        return self.forward_no_update(x)

class PersistentMemory(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(num_tokens, dim))

    def forward(self, bsz):
        return self.mem.unsqueeze(0).expand(bsz, -1, -1)

class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        h = self.num_heads
        q = self.wq(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        k = self.wk(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        v = self.wv(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        out_chunks = []
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            q_i = q[:, :, i:i+1, :]
            k_i = k[:, :, start:i+1, :]
            v_i = v[:, :, start:i+1, :]
            logits = torch.matmul(q_i, k_i.transpose(-1, -2)) / math.sqrt(d//h)
            attn = F.softmax(logits, dim=-1)
            val = torch.matmul(attn, v_i)
            out_chunks.append(val)
        out = torch.cat(out_chunks, dim=2).transpose(1,2).reshape(bsz, seq_len, d)
        out = self.out(out)
        return out

class FullAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        h = self.num_heads
        q = self.wq(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        k = self.wk(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        v = self.wv(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d//h)
        attn = F.softmax(logits, dim=-1)
        val = torch.matmul(attn, v)
        out = val.transpose(1,2).reshape(bsz, seq_len, d)
        out = self.out(out)
        return out

class TitanMAC(nn.Module):
    def __init__(self, dim, chunk_size, hidden_dim, memory_depth, num_persistent_tokens):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.ltm = DeepMemoryModule(dim, hidden_dim, dim, memory_depth)
        self.pm = PersistentMemory(num_persistent_tokens, dim)
        self.attn = FullAttention(dim, 4)

    def segment(self, x):
        bsz, seq_len, d = x.size()
        chunks = []
        idx = 0
        while idx < seq_len:
            chunks.append(x[:, idx:idx+self.chunk_size, :])
            idx += self.chunk_size
        return chunks

    def forward(self, x):
        bsz, seq_len, d = x.size()
        out_chunks = []
        segs = self.segment(x)
        for sg in segs:
            q2d = sg.reshape(-1, d)
            with torch.no_grad():
                ret2d = self.ltm.forward_no_update(q2d)
            ret = ret2d.view(sg.size())
            p = self.pm(bsz)
            cat_in = torch.cat([p, ret, sg], dim=1)
            out_attn = self.attn(cat_in)
            chunk_out = out_attn[:, -sg.size(1):, :]
            k2d = sg.reshape(-1, d)
            v2d = chunk_out.reshape(-1, d)
            self.ltm.forward_with_update(k2d, v2d)
            out_chunks.append(chunk_out)
        return torch.cat(out_chunks, dim=1)

class TitanMAG(nn.Module):
    def __init__(self, dim, hidden_dim, memory_depth, num_persistent_tokens, window_size):
        super().__init__()
        self.dim = dim
        self.pm = PersistentMemory(num_persistent_tokens, dim)
        self.swa = SlidingWindowAttention(dim, 4, window_size)
        self.ltm = DeepMemoryModule(dim, hidden_dim, dim, memory_depth)
        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        pm_out = self.pm(bsz)
        x_in = torch.cat([pm_out, x], dim=1)
        out_swa = self.swa(x_in)
        ltm_in = x_in.reshape(-1, d)
        self.ltm.forward_with_update(ltm_in, out_swa.reshape(-1, d))
        out_mem = self.ltm.forward_no_update(ltm_in).view(*x_in.shape)
        g1 = torch.sigmoid(self.lin1(out_swa))
        g2 = torch.sigmoid(self.lin2(out_mem))
        out_merged = g1*out_swa + g2*out_mem
        return out_merged[:, pm_out.size(1):, :]

class TitanMAL(nn.Module):
    def __init__(self, dim, hidden_dim, memory_depth, num_persistent_tokens, window_size):
        super().__init__()
        self.pm = PersistentMemory(num_persistent_tokens, dim)
        self.ltm = DeepMemoryModule(dim, hidden_dim, dim, memory_depth)
        self.swa = SlidingWindowAttention(dim, 4, window_size)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        pm_out = self.pm(bsz)
        x_cat = torch.cat([pm_out, x], dim=1)
        out_ltm = []
        for i in range(x_cat.size(1)):
            token = x_cat[:, i:i+1, :]
            k2d = token.view(-1, d)
            pred = self.ltm.forward_no_update(k2d)
            self.ltm.forward_with_update(k2d, pred)
            out_ltm.append(pred.view(bsz, 1, d))
        out_ltm = torch.cat(out_ltm, dim=1)
        out_swa = self.swa(out_ltm)
        return out_swa[:, pm_out.size(1):, :]

class TitanDataset(Dataset):
    def __init__(self, split, tokenizer_name, seq_len=1024):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.seq_len = seq_len
        self.samples = []
        if split=="train":
            pass
        else:
            pass
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        return torch.randint(0, 32000, (self.seq_len,)), torch.randint(0, 32000, (self.seq_len,))

def titan_collate(batch):
        xs, ys = [], []
        for x,y in batch:
            xs.append(x)
            ys.append(y)
        return torch.stack(xs,dim=0), torch.stack(ys,dim=0)

class TitanModelForLM(nn.Module):
    def __init__(self, titan_module, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.titan = titan_module
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)
        y = self.titan(x)
        logits = self.head(y)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        return logits, None

def get_ds_config(args):
    return {
        "train_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": args.grad_acc,
        "fp16": {
            "enabled": args.fp16
        },
        "zero_optimization": {
            "stage": args.zero_stage
        },
        "zero_allow_untested_optimizer": True
    }

def train_main(args):
    dist.init_process_group("nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    train_ds = TitanDataset("train", args.tokenizer_name, seq_len=args.seq_len)
    val_ds = TitanDataset("val", args.tokenizer_name, seq_len=args.seq_len)
    train_sampler = RandomSampler(train_ds) if not dist.is_initialized() else torch.utils.data.distributed.DistributedSampler(train_ds)
    val_sampler = SequentialSampler(val_ds)
    train_dl = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, collate_fn=titan_collate, drop_last=True, num_workers=2)
    val_dl = DataLoader(val_ds, sampler=val_sampler, batch_size=args.batch_size, collate_fn=titan_collate, drop_last=False, num_workers=2)

    if args.model_variant=="MAC":
        titan_module = TitanMAC(args.dim, args.chunk_size, args.hidden_dim, args.memory_depth, args.num_persistent_tokens)
    elif args.model_variant=="MAG":
        titan_module = TitanMAG(args.dim, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, args.window_size)
    else:
        titan_module = TitanMAL(args.dim, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, args.window_size)
    model = TitanModelForLM(titan_module, args.vocab_size, args.dim)

    ds_config = get_ds_config(args)
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    global_steps = 0
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            inp, lab = batch
            inp = inp.cuda()
            lab = lab.cuda()
            logits, loss = engine(inp, lab)
            engine.backward(loss)
            engine.step()
            global_steps += 1
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_count = 0
            for batch in val_dl:
                inp, lab = batch
                inp = inp.cuda()
                lab = lab.cuda()
                logits, loss = engine(inp, lab)
                total_loss += loss.item() * inp.size(0)
                total_count += inp.size(0)
            val_ppl = math.exp(total_loss/total_count)
    dist.destroy_process_group()

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--memory_depth", type=int, default=3)
    parser.add_argument("--num_persistent_tokens", type=int, default=4)
    parser.add_argument("--model_variant", type=str, default="MAC")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    train_main(args)
