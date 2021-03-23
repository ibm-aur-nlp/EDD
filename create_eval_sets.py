import json
import argparse
import os
from PIL import Image
from tqdm import tqdm
from html import escape

parser = argparse.ArgumentParser(description='Prepares data files for training EDD')
parser.add_argument('--annotation', type=str, help='path to annotation file')
parser.add_argument('--image_dir', type=str, help='path to image folder')
parser.add_argument('--out_dir', type=str, help='path to folder to save data files')
args = parser.parse_args()

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
    return HTML, tag_len, cell_len_max


with open(args.annotation, 'r') as fp:
    samples = [img for img in json.load(fp)['images'] if img['split'] in ('val', 'test')]

gt = dict()
for img in tqdm(samples):
    HTML, tag_len, cell_len_max = format_html(img)
    with Image.open(os.path.join(args.image_dir, img['filename'])) as im:
        gt[img['filename']] = {
            'html': HTML,
            'tag_len': tag_len,
            'cell_len_max': cell_len_max,
            'width': im.width,
            'height': im.height,
            'type': 'complex' if '>' in img['html']['structure']['tokens'] else 'simple'
        }

# Save ground truth html of validation set
with open(args.out_dir, 'w') as j:
    json.dump(gt, j)
