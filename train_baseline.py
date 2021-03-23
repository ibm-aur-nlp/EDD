'''
Trains baseline encoder-decoder model (WYGIWYS, Dent et al. 2017)
'''
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from models import Encoder, DecoderWithAttention, DecoderWithAttentionAndLanguageModel
from datasets import TableDatasetEvenLength
import json
import os
from utils import *
import argparse
import sys
from glob import glob
import time

def create_model():
    encoder = Encoder(args.encoded_image_size,
                      use_RNN=args.use_RNN,
                      rnn_size=args.encoder_RNN_size,
                      last_layer_stride=args.cnn_stride if isinstance(args.cnn_stride, int) else None)
    encoder.fine_tune(args.fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=args.encoder_lr) if args.fine_tune_encoder else None

    if args.decoder_type == 1:
        decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                       embed_dim=args.emb_dim,
                                       decoder_dim=args.decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim=args.encoder_RNN_size if args.use_RNN else 512,
                                       dropout=args.dropout)
    elif args.decoder_type == 2:
        decoder = DecoderWithAttentionAndLanguageModel(attention_dim=args.attention_dim,
                                                       embed_dim=args.emb_dim,
                                                       language_dim=args.language_dim,
                                                       decoder_dim=args.decoder_dim,
                                                       vocab_size=len(word_map),
                                                       encoder_dim=args.encoder_RNN_size if args.use_RNN else 512,
                                                       dropout=args.dropout)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)

    return encoder, decoder, encoder_optimizer, decoder_optimizer

def load_checkpoint(checkpoint):
    # Wait until model file exists
    if not os.path.isfile(checkpoint):
        while not os.path.isfile(checkpoint):
            print('Model not found, retry in 10 minutes', file=sys.stderr)
            sys.stderr.flush()
            time.sleep(600)
        # Make sure model file is saved completely
        time.sleep(10)

    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1

    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    encoder.fine_tune(args.fine_tune_encoder)
    if args.fine_tune_encoder:
        if encoder_optimizer is None:
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)
        elif encoder_optimizer.param_groups[0]['lr'] != args.encoder_lr:
            change_learning_rate(encoder_optimizer, args.encoder_lr)
            print('Encoder LR changed to %f' % args.encoder_lr, file=sys.stderr)
            sys.stderr.flush()

    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    if decoder_optimizer is None:
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
    elif decoder_optimizer.param_groups[0]['lr'] != args.decoder_lr:
        change_learning_rate(decoder_optimizer, args.decoder_lr)
        print('Decoder LR changed to %f' % args.decoder_lr, file=sys.stderr)
        sys.stderr.flush()

    return start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline table2html model')
    parser.add_argument('--cnn_stride', default=2, type=json.loads, help='stride for last CNN layer in encoder')
    parser.add_argument('--emb_dim', default=16, type=int, help='embedding dimension')
    parser.add_argument('--encoded_image_size', default=14, type=int, help='encoded image size')
    parser.add_argument('--attention_dim', default=512, type=int, help='attention dimension')
    parser.add_argument('--language_dim', default=512, type=int, help='language model dimension')
    parser.add_argument('--decoder_dim', default=512, type=int, help='decoder dimension')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    parser.add_argument('--epochs', default=10, type=int, help='epochs to train')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--bptt', default=200, type=int, help='TBPTT length')
    parser.add_argument('--encoder_lr', default=0.001, type=float, help='encoder learning rate')
    parser.add_argument('--decoder_lr', default=0.001, type=float, help='decoder learning rate')
    parser.add_argument('--grad_clip', default=5., type=float, help='clip gradients at an absolute value')
    parser.add_argument('--alpha_c', default=1., type=float, help='regularization parameter for doubly stochastic attention')
    parser.add_argument('--print_freq', default=100, type=int, help='verbose frequency')
    parser.add_argument('--fine_tune_encoder', dest='fine_tune_encoder', action='store_true', help='fine-tune encoder')
    parser.add_argument('--use_RNN', dest='use_RNN', action='store_true', help='transform image features with LSTM')
    parser.add_argument('--decoder_type', default=1, type=int, help='Type of decoder (1: baseline, 2: with LM)')
    parser.add_argument('--encoder_RNN_size', default=512, type=int, help='LSTM size for the encoder')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--data_folder', default='data/pubmed_dual', type=str, help='path to folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='pubmed_300_max_tag_len_100_max_cell_len_512_max_image_size', type=str, help='base name shared by data files')
    parser.add_argument('--out_dir', type=str, help='path to save checkpoints')
    parser.add_argument('--keep_AR', dest='keep_AR', action='store_true', help='Keep aspect ratio when resizing input images')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from latest checkpoint if exists')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_combined_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if args.resume:
        existing_ckps = glob(os.path.join(args.out_dir, args.data_name, 'checkpoint_*.pth.tar'))
        existing_ckps = [ckp for ckp in existing_ckps if len(os.path.basename(ckp).split('_')) == 2]
        if existing_ckps:
            existing_ckps = sorted(existing_ckps, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1]))
            latest_ckp = existing_ckps[-1]
            if args.checkpoint is not None:
                latest_epoch = int(os.path.basename(latest_ckp).split('.')[0].split('_')[1])
                checkpoint_epoch = int(os.path.basename(args.checkpoint).split('.')[0].split('_')[1])
                if latest_epoch > checkpoint_epoch:
                    print('Resume from latest checkpoint: %s' % latest_ckp, file=sys.stderr)
                    start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer = load_checkpoint(latest_ckp)
                else:
                    print('Start from checkpoint: %s' % args.checkpoint, file=sys.stderr)
                    start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer = load_checkpoint(args.checkpoint)
            else:
                print('Resume from latest checkpoint: %s' % latest_ckp, file=sys.stderr)
                start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer = load_checkpoint(latest_ckp)
        elif args.checkpoint is not None:
            start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer = load_checkpoint(args.checkpoint)
        else:
            encoder, decoder, encoder_optimizer, decoder_optimizer = create_model()
            start_epoch = 0
    else:
        if args.checkpoint is not None:
            start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer = load_checkpoint(args.checkpoint)
        else:
            encoder, decoder, encoder_optimizer, decoder_optimizer = create_model()
            start_epoch = 0

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # mean and std of PubMed Central table images
    normalize = transforms.Normalize(mean=[0.94247851, 0.94254675, 0.94292611],
                                     std=[0.17910956, 0.17940403, 0.17931663])

    train_loader = TableDatasetEvenLength(args.data_folder, args.data_name,
                                          batch_size=args.batch_size,
                                          transform=transforms.Compose([normalize]))

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        decoder.train_epoch(train_loader=train_loader,
                            encoder=encoder,
                            criterion=criterion,
                            encoder_optimizer=encoder_optimizer,
                            decoder_optimizer=decoder_optimizer,
                            epoch=epoch,
                            args=args)

        # Save checkpoint
        save_checkpoint(args.out_dir, args.data_name, epoch, encoder, decoder,
                        encoder_optimizer, decoder_optimizer)
