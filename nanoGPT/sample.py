"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time

class RadixNode:
    def __init__(self,token=None):
        self.token = token
        self.children = {}    # token -> RadixNode
        self.is_end = False 
        self.depth = 0
        self.kv_cache = None

class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def insert(self,tokens):
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                child = RadixNode(token=tok)
                child.depth =node.depth +1
                node.children[tok] = child
            node = node.children[tok]
        node.is_end = True
        return node 

# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# start = "Django is framework in python \n"
# num_samples = 10 # number of samples to draw
num_samples = 1
# max_new_tokens = 500 # number of tokens generated in each sample
max_new_tokens = 100
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = 'float32'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

prompts = [
    "Django is a framework in python\n",
    "Django is a framework in python used for\n",
    "Django supports\n"
]
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
# start_ids = encode(start)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
encoded = [encode(p) for p in prompts]
max_len = max(len(e) for e in encoded)

tree = RadixTree()
leaf_nodes = []
for prompt in encoded:
    leaf = tree.insert(prompt)
    leaf_nodes.append(leaf)

x = torch.zeros(len(encoded),max_len,dtype=torch.long,device=device)
lengths= []

for i,e in enumerate(encoded):
    x[i, :len(e)] = torch.tensor(e,device=device)


################ For Printing KV cache sizez ####################
# def print_kv_sizes(node, prefix_len=0):
#     if node.kv_cache is not None:
#         k0, v0 = node.kv_cache[0]
#         if k0 is not None:   # 🔑 ADD THIS CHECK
#             print(f"Prefix length {prefix_len}: KV shape {k0.shape}")
#     for child in node.children.values():
#         print_kv_sizes(child, prefix_len + 1)

# run generation
# def print_tree(node, prefix=[]):
#     if node.is_end:
#         print("Prompt ends at:", prefix)

#     for tok, child in node.children.items():
#         print_tree(child, prefix + [tok])

# print_tree(tree.root)

def get_current_kv_cache(model):
    """
    Returns a list of (k, v) for each transformer layer.
    """
    kv = []
    for block in model.transformer.h:
        kv.append((block.attn.cache_k, block.attn.cache_v))
    return kv

def load_kv_cache(model, kv_cache):
    """
    Loads KV cache into model attention layers.
    """
    for block, (k, v) in zip(model.transformer.h, kv_cache):
        block.attn.cache_k = k
        block.attn.cache_v = v
        block.attn.use_cache = True

def build_kv_cache_for_tree(model, node, parent_kv=None):
    """
    Recursively builds KV cache for radix tree nodes.
    """
    # 1. Load parent KV or reset at root
    if parent_kv is None:
        for block in model.transformer.h:
            block.attn.reset_cache()
            block.attn.use_cache = True
    else:
        load_kv_cache(model, parent_kv)

    # 2. ONLY process token if this node actually has one
    if node.token is not None:
        idx = torch.tensor([[node.token]], dtype=torch.long, device=device)
        _ = model(idx)  # builds KV for this token

    # 3. Store KV cache at this node
    node.kv_cache = get_current_kv_cache(model)

    # 4. Recurse on children
    for child in node.children.values():
        build_kv_cache_for_tree(model, child, node.kv_cache)


build_kv_cache_for_tree(model, tree.root)





# # print_kv_sizes(tree.root)

def run_single_prompt(model, prompt_tokens):
    """
    Runs one prompt end-to-end:
    1. Clears + builds KV cache from prompt
    2. Generates continuation
    """
    # Phase A: build KV cache from prompt
    process_prompt_only(model, prompt_tokens)

    # Phase B: generate continuation
    last_token = prompt_tokens[-1]
    idx = torch.tensor([[last_token]], device=device)

    y = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_cache=True
    )

    return prompt_tokens + y[0].tolist()


def process_prompt_only(model, prompt_tokens):

    """
    Processes prompt tokens to build KV cache.
    Does NOT generate new tokens.
    """
    # reset KV cache
    for block in model.transformer.h:
        block.attn.reset_cache()
        block.attn.use_cache = True

    # feed full prompt at once
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, :]
    _ = model(idx)

    # KV cache is now populated inside the model

def generate_from_cached_kv(model, leaf_node, prompt_tokens):
    # 1. Load KV cache from radix tree
    load_kv_cache(model, leaf_node.kv_cache)

    # 2. Start generation from last prompt token
    last_token = prompt_tokens[-1]
    idx = torch.tensor([[last_token]], device=device)

    y = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_cache=True
    )

    return prompt_tokens + y[0].tolist()



with torch.no_grad():
    start_time = time.time()
    for i, prompt_tokens in enumerate(encoded):
        print(f"\nPrompt {i+1}: {prompts[i].strip()}")

        output_tokens = generate_from_cached_kv(
            model,
            leaf_nodes[i],
            prompt_tokens
        )

        print("OUTPUT:")
        print(decode(output_tokens))
        print("-" * 40)
    end_time = time.time()
    print(f"Time required For Caching: {end_time - start_time:.3f} sec")




with torch.no_grad():
    start_time = time.time()
    for i, prompt_tokens in enumerate(encoded):
        print(f"\nPrompt {i+1}: {prompts[i].strip()}")

        output_tokens = run_single_prompt(model, prompt_tokens)

        print("OUTPUT:")
        print(decode(output_tokens))
        print("-" * 40)
    end_time = time.time()
    print(f"Time required For Without Caching: {end_time - start_time:.3f} sec")




# with torch.no_grad():
#     start_time = time.time()
#     with ctx:
#         # for k in range(num_samples):
#         #     # model.clear_kv_cache()
#         #     y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,use_cache=True)
#         #     print(decode(y[0].tolist()))
#         #     print('---------------')
#         y = model.generate(
#             x,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k,
#             use_cache=True
#         )
#     end_time = time.time()

#     for i in range(len(prompts)):
#         print(f"PROMPT {i+1}: {prompts[i].strip()}")
#         print("OUTPUT:")
#         print(decode(y[i].tolist()))
#         print("-" * 40)

#     print(f"Time required: {end_time - start_time:.3f} sec")


