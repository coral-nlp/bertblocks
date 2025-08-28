import torch
from transformers import AutoTokenizer

import bertblocks as bb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
seq = tokenizer(
    [
        "The cat sat on the mat.",
        "He didn't know why she left.",
        "Can you believe it's already August?",
        "Running late, she skipped breakfast.",
        "Wow, that's an incredibly fast response!",
    ],
    return_tensors="pt",
    padding="max_length",
).to(device)

model = bb.from_huggingface(model_name, load_weights=True, add_pooling_layer=False).to(device)

out = model(seq["input_ids"], attention_mask=seq["attention_mask"])

print(out.last_hidden_state)
