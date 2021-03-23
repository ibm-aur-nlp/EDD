import torch
import numpy as np
import json
import torchvision.transforms as transforms
import skimage.transform
import argparse
from PIL import Image, ImageDraw, ImageFont
from utils import image_rescale, image_resize
from metric import format_html
import os
from glob import glob
from tqdm import tqdm
import shutil
import textwrap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, image_path, word_map,
                              image_size=448, max_steps=400, beam_size=3,
                              vis_att=False):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
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

    return decoder.inference(encoder_out, word_map, max_steps, beam_size, return_attention=vis_att)

def visualize_result(image_path, res, rev_word_map, smooth=True, image_size=448):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param res: result of inference model
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """

    def vis_attention(c, image, alpha, smooth, image_size, original_size, x_offset=0, cap=None):
        alpha = np.array(alpha)
        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha, upscale=image_size / alpha.shape[0], sigma=4)
        else:
            alpha = alpha.repeat(image_size / alpha.shape[0], axis=0).repeat(image_size / alpha.shape[0], axis=1)
        if cap is None:
            alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
        else:
            alpha *= 1 / cap
            alpha[alpha > 1.] = 1.
        alpha *= 255.
        alpha = alpha.astype('uint8')
        alpha = Image.fromarray(alpha)
        image = image.convert("RGBA")
        alpha = alpha.convert("RGBA")
        new_img = Image.blend(image, alpha, 0.6)
        new_img = new_img.resize(original_size, Image.LANCZOS)
        if c:
            font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 24)
            # font = ImageFont.truetype(os.environ["DATA_DIR"] + "/Table2HTML/dejavu/DejaVuSansMono-Bold.ttf", 24)
            lines = textwrap.wrap(c, width=25)
            w, h = font.getsize(lines[0])
            H = h * len(lines)
            y_text = original_size[1] / 2 - H / 2
            draw = ImageDraw.Draw(new_img)
            for line in lines:
                w, h = font.getsize(line)
                draw.text(((original_size[0] - w) / 2 + x_offset, y_text), line, (255, 255, 255), font=font)
                y_text += h
        return new_img

    if len(res) == 2:
        tags, cells = res
    elif len(res) == 4:
        tags, tag_alphas, cells, cell_alphas = res
    with open(image_path.replace('.png', '.html'), 'w') as fp:
        fp.write(format_html(tags, rev_word_map['tag'], cells, rev_word_map['cell']))

    if len(res) == 4:
        image, original_size = image_resize(image_path, image_size, False)
        folder = image_path[:-4]
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        os.makedirs(os.path.join(folder, 'structure'))
        os.makedirs(os.path.join(folder, 'cells'))

        for ind, (c, alpha) in enumerate(zip(tags[1:], tag_alphas[1:]), 1):
            if ind <= 50 or len(tags[1:]) - ind <= 50:
                new_img = vis_attention(rev_word_map['tag'][c], image, alpha, smooth, image_size, original_size, cap=None)
                new_img.save(os.path.join(folder, 'structure', '%03d.png' % (ind)), "PNG")

        for j, (cell, alphas) in enumerate(zip(cells, cell_alphas)):
            if cell is not None:
                # for ind, (c, alpha) in enumerate(zip(cell[1:], alphas[1:]), 1):
                #     # if ind <= 5 or len(cell[1:]) - ind <= 5:
                #     new_img = vis_attention(rev_word_map['cell'][c], image, alpha, smooth, image_size, original_size)
                #     new_img.save(os.path.join(folder, 'cells', '%03d_%03d.png' % (j, ind)), "PNG")
                new_img = vis_attention(''.join([rev_word_map['cell'][c] for c in cell[1:-1]]),
                                        image,
                                        np.mean(alphas[1:-1], axis=0) if len(alphas[1:-1]) else np.mean(alphas[1:], axis=0),
                                        smooth, image_size, original_size,
                                        x_offset=50 if j % 3 == 0 and j > 0 else 0,
                                        cap=None)
                new_img.save(os.path.join(folder, 'cells', '%03d.png' % (j)), "PNG")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on given images')

    parser.add_argument('--input', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--image_size', '-is', default=448, type=int, help='target size of image rescaling')
    parser.add_argument('--beam_size', '-b', default={"tag": 3, "cell": 3}, type=json.loads, help='beam size for beam search')
    parser.add_argument('--max_steps', '-ms', default=400, type=json.loads, help='max output steps of decoder')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--vis_attention', dest='vis_attention', action='store_true', help='visualize attention')

    args = parser.parse_args()

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
        rev_word_map = {'tag': {v: k for k, v in word_map['word_map_tag'].items()},
                        'cell': {v: k for k, v in word_map['word_map_cell'].items()}}

    if os.path.isfile(args.input):
        # Encode, decode with attention and beam search
        res = caption_image_beam_search(encoder, decoder, args.input, word_map, args.image_size, args.max_steps, args.beam_size, args.vis_attention)
        if res is None:
            print('No complete sequence is generated')
        else:
            # Visualize caption and attention of best sequence
            visualize_result(args.input, res, rev_word_map, args.smooth, args.image_size)
    elif os.path.exists(args.input):
        images = glob(os.path.join(args.input, '*.png')) + glob(os.path.join(args.input, '*.jpg'))
        for image in tqdm(images):
            # Encode, decode with attention and beam search
            try:
                res = caption_image_beam_search(encoder, decoder, image, word_map, args.image_size, args.max_steps, args.beam_size, args.vis_attention)
            except Exception as e:
                print(e)
                res = None
            if res is not None:
                # Visualize caption and attention of best sequence
                visualize_result(image, res, rev_word_map, args.smooth, args.image_size)
