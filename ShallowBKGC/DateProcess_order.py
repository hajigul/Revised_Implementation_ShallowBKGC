import json
import sys

# Get dataset name from command line or default to FB15K
if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "FB15K"

print(f"=== Starting DateProcess_order.py for {dataset} ===")

json_file = f"{dataset}entityIDx_json"
entity_file = f"{dataset}entity.txt"
order_file = f"{dataset}ent2textOrders.txt"
text_file = f"KGs/{dataset}/entity2text.txt"

# Load entity ID mapping
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        entitydict_data = json.load(f)
    print(f" Loaded entity ID mapping with {len(entitydict_data)} entities")
except FileNotFoundError:
    print(f" Error: {json_file} not found!")
    print(f"Please run: python main.py --dataset KGs/{dataset} --num_of_epochs 1")
    exit(1)

# Write ordered entity list
with open(entity_file, 'w', encoding='utf-8') as f:
    for key in entitydict_data.keys():
        f.write(key + '\n')

print(f" Created {entity_file}")

# Load entity2text
dic = {}
try:
    with open(text_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if '\t' in line:
                b = line.split('\t', 1)
                if len(b) == 2:
                    dic[b[0]] = b[1]
    print(f" Loaded entity2text mapping with {len(dic)} entries")
except FileNotFoundError:
    print(f"  Warning: {text_file} not found. Using entity IDs as text.")
    for key in entitydict_data.keys():
        dic[key] = key

# Create ordered text file
with open(order_file, 'w', encoding='utf-8') as file_EntityText:
    with open(entity_file, 'r', encoding='utf-8') as forder:
        for l in forder:
            eid = l.strip()
            value = dic.get(eid, "unknown entity")
            file_EntityText.write(value + '\n')

print(f" Successfully created '{order_file}'")
print(f"Total entities written: {len(entitydict_data)}")
print(f"\n Next: python DateProcess_npy.py {dataset}")