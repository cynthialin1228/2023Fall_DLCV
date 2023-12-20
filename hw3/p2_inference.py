import os
import json
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from tokenizer import BPETokenizer
import math
import collections
from torch import Tensor
import torch.nn.functional as F
import argparse


tokenizer = BPETokenizer("./encoder.json", "vocab.bpe")

def get_args():
    parser = argparse.ArgumentParser(
        description="vit"
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, default="./hw3/p2_data/images/val", help="path to testing images"
    )
    parser.add_argument(
        "--output_json", "-o", type=str, default="./pred.json", help="output prediction"
    )
    parser.add_argument(
        "--decoder_weights", "-dw", type=str, default="./hw3/p2_data/decoder_model.bin", help="model of svhn"
    )
    parser.add_argument(
        "--model_path", "-um", type=str, default="./p2_model.bin", help="model weights of cross_attn"
    )
    return parser.parse_args()

class InferenceDataset:
    def __init__(self, data_dir):
        super(InferenceDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname)).convert('RGB')
        img = self.transform(img)
        return img, fname


class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.n_embd
        self.num_heads = cfg.n_head
        self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        # self.q = lora.Linear(self.embed_dim, self.embed_dim, r=16)
        # self.k = lora.Linear(self.embed_dim, self.embed_dim, r=16)
        # self.v = lora.Linear(self.embed_dim, self.embed_dim, r=16)
        # self.c_proj = lora.Linear(self.embed_dim, self.embed_dim, r=16)

        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_dim ** -0.5
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, context):
        b, t, _ = x.size()
        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        # print(f"attn_scores.shape: {attn_scores.shape}")
        # print((self.bias[:, :, :t, :t] == 0).shape)
        # attn_scores = attn_scores.masked_fill(self.bias[:, :, :t, :t] == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        # print(f"attn_probs.shape: {attn_probs.shape}")

        return self.c_proj((attn_probs @ v).transpose(1, 2).contiguous().view(b, t, self.embed_dim)), attn_probs


class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=32)
        # self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=16)
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)

        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            # ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=32)),
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            # ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=32)),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
        # only add lora in the last few mlp blocks

    def forward(self, x, context):
        x = x + self.attn(self.ln_1(x))
        # x = x + self.cross_attn(self.ln_1(x), context)
        # x = x + self.mlp(self.ln_2(x))
        x0, attn_prob = self.cross_attn(self.ln_2(x), context)
        x = x + x0
        x = x + self.mlp(self.ln_3(x))
        return x, attn_prob

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        # self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, r=16)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, encoder_output: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        # x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
        attn_probs = 0
        for block in self.transformer.h:
            x0, attn_probs = block(x, encoder_output)  # Pass both x and encoder_output to each block
            x = x0

        x = self.lm_head(self.transformer.ln_f(x))
        # x = self.lm_head(self.transformer.ln_f(self.transformer.h(x, encoder_output)))
        return x, attn_probs
class VisionLanguageModel(nn.Module):
    def __init__(self, decoder_cfg):
        super(VisionLanguageModel, self).__init__()
        self.vit = timm.create_model('vit_large_patch14_224_clip_laion2b', pretrained=True, num_classes=0)
        self.decoder = Decoder(decoder_cfg)
        self.output_reshape = nn.Linear(1024, 768)

    def forward(self, images, input_ids):
        x = self.vit.forward_features(images)
        x = self.output_reshape(x)
        decoder_output, attn_probs = self.decoder(input_ids, x)
        return decoder_output, attn_probs



def generate(model, image, sequence, max_tokens=70, temperature=1.0):
    for _ in range(max_tokens):
        out, attn_prob = model.forward(image, sequence)
        out = out[:, -1, :] / temperature
        probs = F.softmax(out, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        sequence = torch.cat([sequence, next_token], dim=1)
        if next_token.item() == 50256:
            break
    attn_prob = attn_prob[:, -1, -1, 1:]  # Reshape to [1, 1, 1, 256]
    return sequence.cpu().flatten(), attn_prob



def infer(input_dir, output_json, decoder_weights, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    config = Config(checkpoint = decoder_weights)
    model = VisionLanguageModel(config).to(device)
    model.load_state_dict(torch.load(model_path), strict=False)

    test_dataset = InferenceDataset(input_dir)
    print("test_dataset size:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=16)


    model.eval()
    initial_sequence = torch.tensor([50256]).unsqueeze(0).to(device)
    predictions ={}
    for images, filenames in tqdm(test_loader):
        images = images.to(device)

        for image, filename in zip(images, filenames):
            sequence, _ = generate(model, image.unsqueeze(0), initial_sequence)
            tts = sequence.tolist()[1:]
            if 50256 in tts:
                tts = tts[:tts.index(50256)]
            decoded_sequence = tokenizer.decode(tts) 
            
            filename = filename.split('.')[0]
            predictions[filename] = decoded_sequence

    with open(output_json, 'w') as f:
        json.dump(predictions, f, indent=4)

    return

if __name__ == "__main__":
    args = get_args()
    infer(args.input_dir, args.output_json, args.decoder_weights, args.model_path)