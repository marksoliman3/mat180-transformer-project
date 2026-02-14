import os
import urllib.request
import torch

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_DATA_FILE = os.path.join(_DATA_DIR, "input.txt")

# The raw file URL from Karpathy's char-rnn repository on GitHub
_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# Download Tiny Shakespeare (if needed) and return the raw text
def load_corpus():

    # Create the data/ directory if it doesn't exist yet.
    os.makedirs(_DATA_DIR, exist_ok=True)

    # Only download if the file is missing — avoids re-downloading every time you run the script
    if not os.path.exists(_DATA_FILE):
        print(f"Downloading Tiny Shakespeare to {_DATA_FILE} ...")
        urllib.request.urlretrieve(_DATA_URL, _DATA_FILE)
        print("Download complete.")
    else:
        print(f"Corpus already exists at {_DATA_FILE}, skipping download.")

    with open(_DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def build_tokenizer(text):
    
    # set(text) extracts every unique character in the corpus, sorted() puts them in a consistent order
    chars = sorted(set(text))

    stoi = {ch: i for i, ch in enumerate(chars)} # stoi lets us go from character → integer  (used by encode)
    itos = {i: ch for i, ch in enumerate(chars)} # itos lets us go from integer → character  (used by decode)

    vocab_size = len(chars)

    return stoi, itos, vocab_size


def encode(s, stoi):

    return [stoi[ch] for ch in s]


def decode(tokens, itos):

    return "".join(itos[i] for i in tokens)



def split_data( text, stoi, train_fraction):
   
    # Step A: Encode the entire corpus into a list of integers
    token_ids = encode(text, stoi)

    # Step B: Convert the Python list into a PyTorch tensor
    data = torch.tensor(token_ids, dtype=torch.long)

    # Step C: Split into train and val by computing the cutoff index.
    split_idx = int(len(data) * train_fraction)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data


block_size: int = 128   # context window length (matches max_seq_len in model.py)
batch_size: int = 32    # number of parallel sequences per batch


def get_batch(split_data):
   
    # Generate batch_size random starting indices
    ix = torch.randint(len(split_data) - block_size, (batch_size,))

    # For each random starting index, slice out a sequence of block_size tokens for x, and the same sequence shifted right by 1 for y.
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])

    return x, y