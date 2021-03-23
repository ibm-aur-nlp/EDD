'''
Combines structual tokens and cell tokens into one sequence to train baseline
encoder-decoder model (WYGIWYS, Dent et al. 2017)
'''
import json
from collections import Counter
from tqdm import tqdm

max_len = 0  # 4191
for subset in ['TRAIN']:
    with open('data/pubmed_dual/%s_TAGLENS_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json' % subset, 'r') as fp:
        taglens = json.load(fp)
    with open('data/pubmed_dual/%s_CELLLENS_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json' % subset, 'r') as fp:
        celllens = json.load(fp)

    total_lens = []
    for taglen, celllen in zip(taglens, celllens):
        total_len = taglen
        for l in celllen:
            total_len += l - 2
        total_lens.append(total_len)
    with open('data/pubmed_dual/%s_TABLELENS_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json' % subset, 'w') as fp:
        json.dump(total_lens, fp)
    max_len = max(max_len, max(total_lens))
print(max_len)

with open('data/pubmed_dual/TRAIN_TAGS_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json', 'r') as fp:
    tags = json.load(fp)
with open('data/pubmed_dual/TRAIN_TAGLENS_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json', 'r') as fp:
    taglens = json.load(fp)
with open('data/pubmed_dual/TRAIN_CELLS_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json', 'r') as fp:
    cells = json.load(fp)
with open('data/pubmed_dual/TRAIN_CELLLENS_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json', 'r') as fp:
    celllens = json.load(fp)
with open('data/pubmed_dual/WORDMAP_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json', 'r') as fp:
    word_map = json.load(fp)
    rev_word_map = {'tag': {v: k for k, v in word_map['word_map_tag'].items()},
                    'cell': {v: k for k, v in word_map['word_map_cell'].items()}}

tables = []

for tag, taglen, cell, celllen in tqdm(zip(tags, taglens, cells, celllens)):
    table = [rev_word_map['tag'][ind] for ind in tag[1:taglen - 1]]
    offset = 0
    j = 0
    while j < len(cell):
        ins = table[offset:].index('</td>')
        c = [rev_word_map['cell'][ind] for ind in cell[j][1:celllen[j] - 1]]
        table = table[:offset + ins] + c + table[offset + ins:]
        offset += ins + len(c) + 1
        j += 1
    tables.append(table)

word_freq = Counter()
for table in tqdm(tables):
    word_freq.update(table)
words = [w for w in word_freq.keys() if word_freq[w] > 5]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

for i in tqdm(range(len(tables))):
    tables[i] = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in tables[i]] + [
        word_map['<end>']] + [word_map['<pad>']] * (max_len - 2 - len(tables[i]))

with open('data/pubmed_dual/WORDMAP_combined_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json', 'w') as fp:
    json.dump(word_map, fp)

with open('data/pubmed_dual/TRAIN_TABLES_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json', 'w') as fp:
    json.dump(tables, fp)
