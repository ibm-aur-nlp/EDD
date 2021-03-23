'''
Visualizes 2D TSNE projection of the embedding of structural tokens and
cell tokens of a pre-trained EDD model.

Cell tokens are categorized according to
http://www.unicode.org/reports/tr44/#General_Category_Values
'''
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
import csv
import sys
import argparse

def tsne_plot(labels, tokens, ax):
    "Creates and TSNE model and plots it"
    tsne_model = TSNE(perplexity=3, learning_rate=100.0, n_components=2, init='pca', n_iter=2000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    scatter = ax.scatter(x, y, c=colors, cmap=plt.get_cmap('tab10'))
    handles, _ = scatter.legend_elements()
    if task == 'word_map_cell':
        handles[7], handles[9] = handles[9], handles[7]
    legend1 = ax.legend(handles, cats,
                        loc=0, title="Categories of tokens")
    ax.add_artist(legend1)
    for i in range(len(x)):
        ax.annotate(labels[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')


parser = argparse.ArgumentParser(
    description='Visualizes 2D TSNE projection of the embedding of structural tokens and cell tokens of a pre-trained EDD model.'
)
parser.add_argument('--word_map', type=str, help='path to word map')
parser.add_argument('--model', type=str, help='path to model')
args = parser.parse_args()


task = 'word_map_%s' % ('tag' if len(sys.argv) == 1 else sys.argv[1])
with open(args.word_map, 'r') as fp:
    word_map = json.load(fp)
    tags = list(word_map['word_map_tag'].keys())
    chars = list(word_map['word_map_cell'].keys())

model = torch.load(args.model, map_location='cpu')
cell_emb = model['decoder'].cell_decoder.embedding.weight.detach().numpy()
tag_emb = model['decoder'].tag_decoder.embedding.weight.detach().numpy()
tag_tokens = [tag_emb[word_map['word_map_tag'][c]] for c in tags]
cell_tokens = [cell_emb[word_map['word_map_cell'][c]] for c in chars]

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# Plot structural tokens
colors = []
cats = ['Tags', 'Colspans', 'Rowspans', 'Auxilary tokens']
for label in tags:
    if 'colspan=' in label:
        colors.append(3)
    elif 'rowspan=' in label:
        colors.append(5)
    elif label in ('<start>', '<end>', '<unk>', '<pad>'):
        colors.append(9)
    else:
        colors.append(0)
tsne_plot(tags, tag_tokens, ax[0])

# Plot cell tokens
with open('data/Categories.txt') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    cat = {}
    for row in reader:
        cat[chr(int(row[0], 16))] = {'group': row[1], 'name': row[-1]}
colors = []
cats = ['Tags', 'Digits', 'Latin letters', 'Greek letters',
        'Currency symbols', 'Math symbols', 'Other symbols',
        'Open or end punctuation', 'Other punctuation',
        'Other']
for label in chars:
    if len(label) > 1:
        colors.append(0)
    else:
        # Number, decimal
        if cat[label]['group'] == 'Nd':
            colors.append(1)
        # Letter (lower or upper case)
        elif cat[label]['group'] in ('Ll', 'Lu'):
            # Latin letters
            if cat[label]['name'].split(' ')[0] == 'LATIN':
                colors.append(2)
            # Greek letters
            elif cat[label]['name'].split(' ')[0] == 'GREEK':
                colors.append(3)
            else:
                colors.append(7)
        elif cat[label]['group'][0] == 'S':
            # Symbol, currency
            if cat[label]['group'][1] == 'c':
                colors.append(4)
            # Symbol, math
            elif cat[label]['group'][1] == 'm':
                colors.append(5)
            # Symbol, other
            else:
                colors.append(6)
        # Punctuation, (open or end)
        elif cat[label]['group'] in ('Ps', 'Pe'):
            colors.append(9)

        elif cat[label]['group'] in ('Po'):
            colors.append(8)
        else:
            colors.append(7)
tsne_plot(chars, cell_tokens, ax[1])

plt.tight_layout()
plt.show()
