import numpy as np
import torch
import sys
from pytorch_pretrained_bert import BertTokenizer, BertModel

# ====================== Get Dataset from Command Line ======================
if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "FB15K"   # Changed to FB15K

print(f"=== Starting BERT Embedding Generation for {dataset} ===")

# Dynamic filenames
order_file = f"{dataset}ent2textOrders.txt"
npy_file = f"{dataset}EntTxtWeights.npy"

# ====================== Load BERT Model ======================
print("Loading BERT model (this may take a moment)...")
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ====================== Helper Functions ======================
def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        _, pooled_output = model(tokens_tensor, segments_tensors)
    pooled_output_emb = torch.squeeze(pooled_output, dim=0)
    return pooled_output_emb.cpu().numpy()   # directly return numpy


# ====================== Main Processing ======================
print(f"Starting BERT embedding generation for {dataset}...")

target_CLS_embeddings = []

try:
    with open(order_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total = len(lines)
    print(f"Found {total} entities to process.")

    for idx, line in enumerate(lines):
        try:
            processed_line = line.strip('\n').strip()
            if not processed_line:
                # Use empty string for empty descriptions
                processed_line = "unknown entity"

            tokens_tensor, segments_tensors = bert_text_preparation(processed_line, tokenizer)
            emb = get_bert_embeddings(tokens_tensor, segments_tensors, model)
            
            target_CLS_embeddings.append(emb)

            if (idx + 1) % 1000 == 0 or (idx + 1) == total:
                print(f"Processed {idx + 1}/{total} entities...")

        except Exception as e:
            print(f"  Warning in line {idx + 1}: {e}")
            print(f"Using zero embedding for: {line.strip()}")
            # Use zero embedding as fallback
            target_CLS_embeddings.append(np.zeros(768))

    # Convert to numpy array and save
    print("Converting to numpy array and saving...")
    target_CLS_embeddings = np.array(target_CLS_embeddings)
    np.save(npy_file, target_CLS_embeddings)

    print(f" Success! Saved '{npy_file}' with shape: {target_CLS_embeddings.shape}")

except FileNotFoundError:
    print(f" Error: File '{order_file}' not found!")
    print(f"Please run: python DateProcess_order.py {dataset} first.")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n Next: python main.py --dataset KGs/{dataset} --num_of_epochs 150")