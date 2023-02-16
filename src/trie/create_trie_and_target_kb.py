import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import json
import argparse
import pickle
import pandas as pd
from transformers import BartTokenizer
from trie import Trie
def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

with open('../benchmarks/bc5cdr/target_kb.json', 'r') as f:
    cui2str = json.load(f)

entities = []
for cui in cui2str:
    entities += cui2str[cui]
#print(entities)
print("-------------------")
print([list(tokenizer(' ' + entity.lower())['input_ids'][1:]) for entity in entities[0:10]])
print("**************************")
trie = Trie([16]+list(tokenizer(' ' + entity.lower())['input_ids'][1:]) for entity in tqdm(entities)).trie_dict
print(trie.get([16]))
with open('../benchmarks/bc5cdr/trie.pkl', 'wb') as w_f:
    pickle.dump(trie, w_f)
print("finish running!")
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
#
# all_country_names = pickle_load_large_file('../benchmarks/lgl/all_country_names.pkl')
#
# print(all_country_names)
# print("-------------------")
# print([list(tokenizer(' ' + entity.lower())['input_ids'][1:]) for entity in all_country_names[0:10]])
# print("**************************")
# trie = Trie([11]+list(tokenizer(' ' + entity.lower())['input_ids'][1:]) for entity in tqdm(all_country_names)).trie_dict
# print(trie.get([11]))
# with open('../benchmarks/lgl/trie.pkl', 'wb') as w_f:
#     pickle.dump(trie, w_f)
# print("finish running!")