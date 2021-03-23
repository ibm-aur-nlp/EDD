'''
Prepares data files for training and validating EDD model. The following files
are generated in <out_dir>:
 - TRAIN_IMAGES_<POSTFIX>.h5          # Training images
 - TRAIN_TAGS_<POSTFIX>.json          # Training structural tokens
 - TRAIN_TAGLENS_<POSTFIX>.json       # Length of training structural tokens
 - TRAIN_CELLS_<POSTFIX>.json         # Training cell tokens
 - TRAIN_CELLLENS_<POSTFIX>.json      # Length of training cell tokens
 - TRAIN_CELLBBOXES_<POSTFIX>.json    # Training cell bboxes
 - VAL.json                           # Validation ground truth
 - WORDMAP_<POSTFIX>.json             # Vocab

<POSTFIX> is formatted according to input args (keep_AR, max_tag_len, ...)
'''
import json
import jsonlines
from tqdm import tqdm
import argparse
from collections import Counter
import os
from PIL import Image
import h5py
import numpy as np
from utils import image_rescale
from html import escape
from lxml import html

def is_valid(img):
    if len(img['html']['structure']['tokens']) > args.max_tag_len:
        return False
    for cell in img['html']['cells']:
        if len(cell['tokens']) > args.max_cell_len:
            return False
    with Image.open(os.path.join(args.image_dir, img['split'], img['filename'])) as im:
        if im.width > args.max_image_size or im.height > args.max_image_size:
            return False
    return True

def scale(bbox, orig_size):
    ''' Normalizes bbox to 0 - 1
    '''
    if bbox[0] == 0:
        return bbox
    else:
        x = float((bbox[3] + bbox[1]) / 2) / orig_size[0]  # x center
        y = float((bbox[4] + bbox[2]) / 2) / orig_size[1]  # y center
        width = float(bbox[3] - bbox[1]) / orig_size[0]
        height = float(bbox[4] - bbox[2]) / orig_size[1]
        return [1, x, y, width, height]

def format_html(img):
    ''' Formats HTML code from tokenized annotation of img
    '''
    tag_len = len(img['html']['structure']['tokens'])
    cell_len_max = max([len(c['tokens']) for c in img['html']['cells']])
    HTML = img['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(HTML) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], img['html']['cells'][::-1]):
        if cell:
            cell = ''.join([escape(token) if len(token) == 1 else token for token in cell['tokens']])
            HTML.insert(i + 1, cell)
    HTML = '<html><body><table>%s</table></body></html>' % ''.join(HTML)
    root = html.fromstring(HTML)
    for td, cell in zip(root.iter('td'), img['html']['cells']):
        if 'bbox' in cell:
            bbox = cell['bbox']
            td.attrib['x'] = str(bbox[0])
            td.attrib['y'] = str(bbox[1])
            td.attrib['width'] = str(bbox[2] - bbox[0])
            td.attrib['height'] = str(bbox[3] - bbox[1])
    HTML = html.tostring(root, encoding='utf-8').decode()
    return HTML, tag_len, cell_len_max


parser = argparse.ArgumentParser(description='Prepares data files for training EDD')
parser.add_argument('--annotation', type=str, help='path to annotation file')
parser.add_argument('--image_dir', type=str, help='path to image folder')
parser.add_argument('--out_dir', type=str, help='path to folder to save data files')
parser.add_argument('--min_word_freq', default=5, type=int, help='minimium frequency for a token to be included in vocab')
parser.add_argument('--max_tag_len', default=300, type=int, help='maximium number of structural tokens for a sample to be included')
parser.add_argument('--max_cell_len', default=100, type=int, help='maximium number tokens in a cell for a sample to be included')
parser.add_argument('--max_image_size', default=512, type=int, help='maximium image width/height a sample to be included')
parser.add_argument('--image_size', default=448, type=int, help='target image rescaling size')
parser.add_argument('--keep_AR', default=False, action='store_true', help='keep aspect ratio and pad with zeros when rescaling images')

args = parser.parse_args()

# Read image paths and captions for each image
dataset = 'PubTabNet'
train_image_paths = []
train_image_tags = []
train_image_cells = []
train_image_cell_bboxes = []
val_gt = dict()
word_freq_tag = Counter()

word_freq_cell = Counter()
with jsonlines.open(args.annotation, 'r') as reader:
    for img in tqdm(reader):
        if img['split'] == 'train':
            if is_valid(img):
                tags = []
                cells = []
                cell_bboxes = []
                word_freq_tag.update(img['html']['structure']['tokens'])
                tags.append(img['html']['structure']['tokens'])
                for cell in img['html']['cells']:
                    word_freq_cell.update(cell['tokens'])
                    cells.append(cell['tokens'])
                    if 'bbox' in cell:
                        cell_bboxes.append([1] + cell['bbox'])
                    else:
                        cell_bboxes.append([0, 0, 0, 0, 0])

                path = os.path.join(args.image_dir, img['split'], img['filename'])

                train_image_paths.append(path)
                train_image_tags.append(tags)
                train_image_cells.append(cells)
                train_image_cell_bboxes.append(cell_bboxes)
        elif img['split'] == 'val':
            HTML, tag_len, cell_len_max = format_html(img)
            with Image.open(os.path.join(args.image_dir, img['split'], img['filename'])) as im:
                val_gt[img['filename']] = {
                    'html': HTML,
                    'tag_len': tag_len,
                    'cell_len_max': cell_len_max,
                    'width': im.width,
                    'height': im.height,
                    'type': 'complex' if '>' in img['html']['structure']['tokens'] else 'simple'
                }


if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# Save ground truth html of validation set
with open(os.path.join(args.out_dir, 'VAL.json'), 'w') as j:
    json.dump(val_gt, j)

# Sanity check
assert len(train_image_paths) == len(train_image_tags)

# Create a base/root name for all output files
base_filename = dataset + '_' + \
    str(args.keep_AR) + '_keep_AR_' + \
    str(args.max_tag_len) + '_max_tag_len_' + \
    str(args.max_cell_len) + '_max_cell_len_' + \
    str(args.max_image_size) + '_max_image_size'

words_tag = [w for w in word_freq_tag.keys() if word_freq_tag[w] >= args.min_word_freq]
words_cell = [w for w in word_freq_cell.keys() if word_freq_cell[w] >= args.min_word_freq]

word_map_tag = {k: v + 1 for v, k in enumerate(words_tag)}
word_map_tag['<unk>'] = len(word_map_tag) + 1
word_map_tag['<start>'] = len(word_map_tag) + 1
word_map_tag['<end>'] = len(word_map_tag) + 1
word_map_tag['<pad>'] = 0

word_map_cell = {k: v + 1 for v, k in enumerate(words_cell)}
word_map_cell['<unk>'] = len(word_map_cell) + 1
word_map_cell['<start>'] = len(word_map_cell) + 1
word_map_cell['<end>'] = len(word_map_cell) + 1
word_map_cell['<pad>'] = 0

# Save word map to a JSON
with open(os.path.join(args.out_dir, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
    json.dump({"word_map_tag": word_map_tag, "word_map_cell": word_map_cell}, j)

with h5py.File(os.path.join(args.out_dir, 'TRAIN_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
    # Create dataset inside HDF5 file to store images
    images = h.create_dataset('images', (len(train_image_paths), 3, args.image_size, args.image_size), dtype='uint8')

    enc_tags = []
    tag_lens = []
    enc_cells = []
    cell_lens = []
    cell_bboxes = []
    for i, path in enumerate(tqdm(train_image_paths)):
        # Read images
        img, orig_size = image_rescale(train_image_paths[i], args.image_size, args.keep_AR, return_size=True)
        assert img.shape == (3, args.image_size, args.image_size)
        assert np.max(img) <= 255

        # Save image to HDF5 file
        images[i] = img

        for tag in train_image_tags[i]:
            # Encode captions
            enc_tag = [word_map_tag['<start>']] + [word_map_tag.get(word, word_map_tag['<unk>']) for word in tag] + \
                      [word_map_tag['<end>']] + [word_map_tag['<pad>']] * (args.max_tag_len - len(tag))
            # Find caption lengths
            tag_len = len(tag) + 2

            enc_tags.append(enc_tag)
            tag_lens.append(tag_len)

        __enc_cell = []
        __cell_len = []
        for cell in train_image_cells[i]:
            # Encode captions
            enc_cell = [word_map_cell['<start>']] + [word_map_cell.get(word, word_map_cell['<unk>']) for word in cell] + \
                       [word_map_cell['<end>']] + [word_map_cell['<pad>']] * (args.max_cell_len - len(cell))
            # Find caption lengths
            cell_len = len(cell) + 2

            __enc_cell.append(enc_cell)
            __cell_len.append(cell_len)
        enc_cells.append(__enc_cell)
        cell_lens.append(__cell_len)

        __cell_bbox = []
        for bbox in train_image_cell_bboxes[i]:
            __cell_bbox.append(scale(bbox, orig_size))
        cell_bboxes.append(__cell_bbox)

    # Save encoded captions and their lengths to JSON files
    with open(os.path.join(args.out_dir, 'TRAIN_TAGS_' + base_filename + '.json'), 'w') as j:
        json.dump(enc_tags, j)

    with open(os.path.join(args.out_dir, 'TRAIN_TAGLENS_' + base_filename + '.json'), 'w') as j:
        json.dump(tag_lens, j)

    with open(os.path.join(args.out_dir, 'TRAIN_CELLS_' + base_filename + '.json'), 'w') as j:
        json.dump(enc_cells, j)

    with open(os.path.join(args.out_dir, 'TRAIN_CELLLENS_' + base_filename + '.json'), 'w') as j:
        json.dump(cell_lens, j)

    with open(os.path.join(args.out_dir, 'TRAIN_CELLBBOXES_' + base_filename + '.json'), 'w') as j:
        json.dump(cell_bboxes, j)
