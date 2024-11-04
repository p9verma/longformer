'''
Basic training loop for a GPT model. 

Implements a few additional things not seen in the homeworks:
> autocast, which will use a more efficient datatype for forward pass
> torch.compile, which speeds things up greatly if your hardware supports it

see the flags section to set these.
'''

import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from warmup_cosine import cosine_with_warmup_lr_scheduler

# use this to designate a GPU, if you have more than one
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # run on the CPU, don't have GPU

# turn on tf32 support, can't hurt
# torch.backends.cuda.matmul.allow_tf32=True
# torch.backends.cudnn.allow_tf32=True


# =============================================================================
# all the settings -------------------------------------

# alternative dtype for forward, set to None to turn off
AUTOCAST_DTYPE = None #torch.float16 using CPU

# use torch compile?
COMPILE = False

# Model architecture
D_MODEL = 512
N_HEADS = 16
LAYERS = 8
VOCAB_SIZE = 10000
SEQ_LEN = 256

# training hyperparams
PEAK_LR = 0.0005
WARMUP_STEPS = 300
BATCH_SIZE = 16
ACCUMULATION = 4
GRAD_CLIP = 1.0

# =============================================================================

torch.manual_seed(0)

# ------------ model ----------------
device = torch.device("cpu") #"cuda")

model = GPTModel(
    d_model=D_MODEL, 
    n_heads=N_HEADS, 
    layers=LAYERS, 
    vocab_size=VOCAB_SIZE, 
    max_seq_len=SEQ_LEN,
    window_size=128,
    global_attn_nodes=[0]
)
## BASELINE SELF ATTENTION MODEL
# model = GPTModel(
#     d_model=512, 
#     n_heads=16, 
#     layers=8, 
#     vocab_size=10000, 
#     max_seq_len=SEQ_LEN,  # Test with 256, 512, 1024
#     window_size=None,
#     global_attn_nodes=None
# )

## LONGFORMER ATTENTION MECHANISM
# model = GPTModel(
#     d_model=512, 
#     n_heads=16, 
#     layers=8, 
#     vocab_size=10000, 
#     max_seq_len=SEQ_LEN,  # Test with 256, 512, 1024
#     window_size=128,
#     global_attn_nodes=[0]
# )
param_count = sum(p.numel() for p in model.parameters())
print("PARAMS:", param_count)

model = model.to(device)

# ------------ dataset ----------------

with open('dataset.npy', 'rb') as f:
    dataset = np.load(f, allow_pickle=True)
print(dataset.shape)

# ------------ some collections ----------------

batch_size = BATCH_SIZE
batches = len(dataset)//batch_size
losses = []
tokens = []
total_tokens = 0

# ------------ loss and opt ----------------

loss_fn = torch.nn.CrossEntropyLoss().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
scheduler = cosine_with_warmup_lr_scheduler(opt, batches, WARMUP_STEPS)
# scaler = torch.cuda.amp.GradScaler()

# ------------ train loop ----------------

st = time.time()
for b in range(batches*1):
    bdx = b%batches

    x = dataset[batch_size*bdx:batch_size*(bdx+1), :]
    x = torch.from_numpy(x).to(device)
    inp = x[:, :-1]
    targ = x[:, 1:]

    # if AUTOCAST_DTYPE is not None:
    #     with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
    y = model(inp)
    y = y.transpose(1,2)
    loss = loss_fn(y, targ)

        # scaler.scale(loss).backward()
    if (b+1)%ACCUMULATION==0:
        # scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        # scaler.step(opt)
        # scaler.update()
        opt.zero_grad(set_to_none=True)

    # else:
    #     y = model(inp)
    #     y = y.transpose(1,2)
    #     loss = loss_fn(y, targ)
    #     loss.backward()

    #     torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    #     if (b+1)%ACCUMULATION==0:
    #         opt.step()
    #         opt.zero_grad(set_to_none=True)

    scheduler.step()
    b += 1

    # loss tracking ===============================================
    losses.append(loss.item())
    total_tokens += batch_size*256
    tokens.append(total_tokens)
    elapsed = (time.time() - st)/60
    tokens_hr = (total_tokens/elapsed)*60
    print(b, total_tokens, loss.item(), elapsed, "minutes", tokens_hr, "tokens/hr")


    if b>200 and b%100==0:
        plt.clf()

        # get rid of really early data
        plot_tokens = tokens[200:]
        plot_losses = losses[200:] 

        # raw train
        plt.plot(plot_tokens, plot_losses, color='b', alpha=0.2)
        
        # smoothed train
        w = 50
        smoothed = np.convolve(plot_losses, np.ones(w), 'valid') / w
        smoothed_x = plot_tokens[-len(smoothed):]
        plt.plot(smoothed_x, smoothed, color='b')

        # save linear
        plt.xlabel("Training Tokens")
        plt.ylabel("Loss")
        plt.savefig("./loss_plot.png")

        # # save weights
        torch.save(model.state_dict(), "./model_weights.pt")