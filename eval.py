from __future__ import print_function
import torch
import json
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
import sys
import time
from utils import image_rescale
from metric import format_html, similarity_eval_html
from lxml import html
import numpy as np
from glob import glob
import traceback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_table_beam_search(encoder, decoder, image_path, word_map, rev_word_map,
                              image_size=448, max_steps=400, beam_size=3,
                              dual_decoder=True):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param max_steps: max numerb of decoding steps
    :param beam_size: number of sequences to consider at each decode-step
    :param dual_decoder: if the model has dual decoders
    :return: HTML code of input table image
    """
    # Read image and process
    img = image_rescale(image_path, image_size, False)
    img = img / 255.
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(mean=[0.94247851, 0.94254675, 0.94292611],
                                     std=[0.17910956, 0.17940403, 0.17931663])
    transform = transforms.Compose([normalize])
    image = transform(img).to(device)  # (3, image_size, image_size)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, image_size, image_size)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    res = decoder.inference(encoder_out, word_map, max_steps, beam_size, return_attention=False)
    if res is not None:
        if dual_decoder:
            if len(res) == 2:
                html_string = format_html(res[0], rev_word_map['tag'], res[1], rev_word_map['cell'])
            else:
                html_string = format_html(res[0], rev_word_map['tag'])
        else:
            html_string = format_html(res, rev_word_map)
    else:
        html_string = ''
    return html_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of table2html conversion models')

    parser.add_argument('--image_folder', type=str, help='path to image folder')
    parser.add_argument('--result_json', type=str, help='path to save results (json)')
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--word_map', help='path to word map JSON')
    parser.add_argument('--gt', default=None, type=str, help='path to ground truth')
    parser.add_argument('--image_size', default=448, type=int, help='target size of image rescaling')
    parser.add_argument('--dual_decoder', default=False, dest='dual_decoder', action='store_true', help='the decoder is a dual decoder')
    parser.add_argument('--beam_size', default={"tag": 3, "cell": 3}, type=json.loads, help='beam size for beam search')
    parser.add_argument('--max_steps', default={"tag": 1800, "cell": 600}, type=json.loads, help='max output steps of decoder')

    args = parser.parse_args()

    # Wait until model file exists
    if not os.path.isfile(args.model):
        while not os.path.isfile(args.model):
            print('Model not found, retry in 10 minutes', file=sys.stderr)
            sys.stderr.flush()
            time.sleep(600)
        # Make sure model file is saved completely
        time.sleep(10)
    # Load model
    checkpoint = torch.load(args.model)

    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)

    if args.dual_decoder:
        rev_word_map = {'tag': {v: k for k, v in word_map['word_map_tag'].items()},
                        'cell': {v: k for k, v in word_map['word_map_cell'].items()}}
    else:
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Load ground truth
    if args.gt is not None:
        with open(args.gt, 'r') as j:
            gt = json.load(j)

    normalize = transforms.Normalize(mean=[0.94247851, 0.94254675, 0.94292611],
                                     std=[0.17910956, 0.17940403, 0.17931663])
    transform = transforms.Compose([normalize])

    if args.gt is None:
        # Ground truth of test set is not provide. To evaluate test performance,
        # Please do not specify the ground truth file, and all png images in
        # image_folderwill be converted. Conversion results are saved in a json,
        # which can be uploaded to our evaluation service (coming soon) for
        # evaluation.
        HTML = dict()
        images = glob(os.path.join(args.image_folder, '*.png'))
        for filename in tqdm(images):
            try:
                html_pred = convert_table_beam_search(
                    encoder, decoder, filename, word_map, rev_word_map,
                    args.image_size, args.max_steps, args.beam_size,
                    args.dual_decoder)
            except Exception as e:
                traceback.print_exc()
                html_pred = ''
            HTML[os.path.basename(filename)] = html_pred
        if not os.path.exists(os.path.dirname(args.result_json)):
            os.makedirs(os.path.dirname(args.result_json))
        with open(args.result_json, 'w') as fp:
            json.dump(HTML, fp)
    else:
        # Ground truth of validation set is provide. Please specify the ground
        # truth file, and the TEDS scores on simple, complex, and all table
        # samples will be computed.
        TEDS = dict()
        for filename, attributes in tqdm(gt.items()):
            try:
                html_pred = convert_table_beam_search(
                    encoder, decoder,
                    os.path.join(args.image_folder, filename),
                    word_map, rev_word_map,
                    args.image_size, args.max_steps, args.beam_size,
                    args.dual_decoder)
                if html_pred:
                    TEDS[filename] = similarity_eval_html(html.fromstring(html_pred), html.fromstring(attributes['html']))
                else:
                    TEDS[filename] = 0.
            except Exception as e:
                traceback.print_exc()
                TEDS[filename] = 0.

        simple = [TEDS[filename] for filename, attributes in gt.items() if attributes['type'] == 'simple']
        complex = [TEDS[filename] for filename, attributes in gt.items() if attributes['type'] == 'complex']
        total = [TEDS[filename] for filename, attributes in gt.items()]

        print('TEDS of %d simple tables: %.3f' % (len(simple), np.mean(simple)))
        print('TEDS of %d complex tables: %.3f' % (len(complex), np.mean(complex)))
        print('TEDS of %d all tables: %.3f' % (len(total), np.mean(total)))

        if not os.path.exists(os.path.dirname(args.result_json)):
            os.makedirs(os.path.dirname(args.result_json))
        with open(args.result_json, 'w') as fp:
            json.dump(TEDS, fp)
