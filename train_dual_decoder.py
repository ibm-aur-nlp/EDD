'''
Trains encoder-dual-decoder model
'''
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from models import Encoder, DualDecoder
from datasets import *
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

    decoder = DualDecoder(tag_attention_dim=args.tag_attention_dim,
                          cell_attention_dim=args.cell_attention_dim,
                          tag_embed_dim=args.tag_embed_dim,
                          cell_embed_dim=args.cell_embed_dim,
                          tag_decoder_dim=args.tag_decoder_dim,
                          language_dim=args.language_dim,
                          cell_decoder_dim=args.cell_decoder_dim,
                          tag_vocab_size=len(word_map['word_map_tag']),
                          cell_vocab_size=len(word_map['word_map_cell']),
                          td_encode=(word_map['word_map_tag']['<td>'], word_map['word_map_tag']['>']),
                          decoder_cell=nn.LSTMCell if args.decoder_cell == 'LSTM' else nn.GRUCell,
                          encoder_dim=512,
                          dropout=args.dropout,
                          cell_decoder_type=args.cell_decoder_type,
                          cnn_layer_stride=args.cnn_stride if isinstance(args.cnn_stride, dict) else None,
                          tag_H_grad=not args.detach,
                          predict_content=args.predict_content,
                          predict_bbox=args.predict_bbox)
    decoder.fine_tune_tag_decoder(args.fine_tune_tag_decoder)
    tag_decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.tag_decoder.parameters()),
                                             lr=args.tag_decoder_lr) if args.fine_tune_tag_decoder else None

    cell_decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.cell_decoder.parameters()),
                                              lr=args.cell_decoder_lr) if args.predict_content else None
    cell_bbox_regressor_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.cell_bbox_regressor.parameters()),
                                                     lr=args.cell_bbox_regressor_lr) if args.predict_bbox else None
    return encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer

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
    decoder.tag_H_grad = not args.detach
    decoder.tag_decoder.tag_H_grad = not args.detach
    decoder.predict_content = args.predict_content
    decoder.predict_bbox = args.predict_bbox

    tag_decoder_optimizer = checkpoint['tag_decoder_optimizer']
    decoder.fine_tune_tag_decoder(args.fine_tune_tag_decoder)
    if args.fine_tune_tag_decoder:
        if tag_decoder_optimizer is None:
            tag_decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.tag_decoder.parameters()),
                                                     lr=args.tag_decoder_lr)
        elif tag_decoder_optimizer.param_groups[0]['lr'] != args.tag_decoder_lr:
            change_learning_rate(tag_decoder_optimizer, args.tag_decoder_lr)
            print('Tag Decoder LR changed to %f' % args.tag_decoder_lr, file=sys.stderr)
            sys.stderr.flush()

    cell_decoder_optimizer = checkpoint['cell_decoder_optimizer']
    if args.predict_content:
        if cell_decoder_optimizer is None:
            cell_decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.cell_decoder.parameters()),
                                                      lr=args.cell_decoder_lr)
        elif cell_decoder_optimizer.param_groups[0]['lr'] != args.cell_decoder_lr:
            change_learning_rate(cell_decoder_optimizer, args.cell_decoder_lr)
            print('Cell Decoder LR changed to %f' % args.cell_decoder_lr, file=sys.stderr)
            sys.stderr.flush()

    cell_bbox_regressor_optimizer = checkpoint['cell_bbox_regressor_optimizer']
    if args.predict_bbox:
        if cell_bbox_regressor_optimizer is None:
            cell_bbox_regressor_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.cell_bbox_regressor.parameters()),
                                                             lr=args.cell_bbox_regressor_lr)
        elif cell_bbox_regressor_optimizer.param_groups[0]['lr'] != args.cell_bbox_regressor_lr:
            change_learning_rate(cell_bbox_regressor_optimizer, args.cell_bbox_regressor_lr)
            print('Cell bbox regressor LR changed to %f' % args.cell_bbox_regressor_lr, file=sys.stderr)
            sys.stderr.flush()

    return start_epoch, encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train encoder-dual-decoder table2html model')
    parser.add_argument('--cnn_stride', default=2, type=json.loads, help='stride for last CNN layer in encoder')
    parser.add_argument('--tag_embed_dim', default=16, type=int, help='embedding dimension')
    parser.add_argument('--cell_embed_dim', default=16, type=int, help='embedding dimension')
    parser.add_argument('--encoded_image_size', default=14, type=int, help='encoded image size')
    parser.add_argument('--tag_attention_dim', default=512, type=int, help='tag attention dimension')
    parser.add_argument('--cell_attention_dim', default=512, type=int, help='tag attention dimension')
    parser.add_argument('--language_dim', default=512, type=int, help='language model dimension')
    parser.add_argument('--tag_decoder_dim', default=512, type=int, help='tag decoder dimension')
    parser.add_argument('--cell_decoder_dim', default=512, type=int, help='cell decoder dimension')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    parser.add_argument('--epochs', default=10, type=int, help='epochs to train')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--encoder_lr', default=0.001, type=float, help='encoder learning rate')
    parser.add_argument('--tag_decoder_lr', default=0.001, type=float, help='tag decoder learning rate')
    parser.add_argument('--cell_decoder_lr', default=0.001, type=float, help='cell decoder learning rate')
    parser.add_argument('--cell_bbox_regressor_lr', default=0.001, type=float, help='cell bbox regressor learning rate')
    parser.add_argument('--grad_clip', default=5., type=float, help='clip gradients at an absolute value')
    parser.add_argument('--alpha_tag', default=0., type=float, help='regularization parameter in tag decoder for doubly stochastic attention')
    parser.add_argument('--alpha_cell', default=0., type=float, help='regularization parameter in cell decoder for doubly stochastic attention')
    parser.add_argument('--tag_loss_weight', default=0.5, type=float, help='weight of tag loss')
    parser.add_argument('--cell_loss_weight', default=0.5, type=float, help='weight of cell content loss')
    parser.add_argument('--cell_bbox_loss_weight', default=0.0, type=float, help='weight of cell bbox loss')
    parser.add_argument('--print_freq', default=100, type=int, help='verbose frequency')
    parser.add_argument('--fine_tune_encoder', dest='fine_tune_encoder', action='store_true', help='fine-tune encoder')
    parser.add_argument('--fine_tune_tag_decoder', dest='fine_tune_tag_decoder', action='store_true', help='fine-tune tag decoder')
    parser.add_argument('--cell_decoder_type', default=1, type=int, help='Type of cell decoder (1: baseline, 2: with LM)')
    parser.add_argument('--decoder_cell', default='LSTM', type=str, help='RNN Cell (LSTM or GRU)')
    parser.add_argument('--use_RNN', dest='use_RNN', action='store_true', help='transform image features with LSTM')
    parser.add_argument('--detach', dest='detach', default=False, action='store_true', help='detach the hidden state between structure and cell decoders')
    parser.add_argument('--encoder_RNN_size', default=512, type=int, help='LSTM size for the encoder')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--data_folder', default='data/pubmed_dual', type=str, help='path to folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='pubmed_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size', type=str, help='base name shared by data files')
    parser.add_argument('--out_dir', type=str, help='path to save checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from latest checkpoint if exists')
    parser.add_argument('--predict_content', dest='predict_content', default=False, action='store_true', help='Predict cell content')
    parser.add_argument('--predict_bbox', dest='predict_bbox', default=False, action='store_true', help='Predict cell bbox')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
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
                    start_epoch, encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer = load_checkpoint(latest_ckp)
                else:
                    print('Start from checkpoint: %s' % args.checkpoint, file=sys.stderr)
                    start_epoch, encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer = load_checkpoint(args.checkpoint)
            else:
                print('Resume from latest checkpoint: %s' % latest_ckp, file=sys.stderr)
                start_epoch, encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer = load_checkpoint(latest_ckp)
        elif args.checkpoint is not None:
            start_epoch, encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer = load_checkpoint(args.checkpoint)
        else:
            encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer = create_model()
            start_epoch = 0
    else:
        if args.checkpoint is not None:
            start_epoch, encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer = load_checkpoint(args.checkpoint)
        else:
            encoder, decoder, encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer = create_model()
            start_epoch = 0

    # Move to GPU, if available
    if torch.cuda.device_count() > 1:
        print('Using %d GPUs' % torch.cuda.device_count(), file=sys.stderr)
        if not hasattr(encoder, 'module'):
            print('Parallelize encoder', file=sys.stderr)
            encoder = MyDataParallel(encoder)
        if not hasattr(decoder.tag_decoder, 'module'):
            print('Parallelize tag decoder', file=sys.stderr)
            decoder.tag_decoder = MyDataParallel(decoder.tag_decoder)
        if not hasattr(decoder.cell_decoder, 'module'):
            print('Parallelize cell decoder', file=sys.stderr)
            decoder.cell_decoder = MyDataParallel(decoder.cell_decoder)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = {'tag': nn.CrossEntropyLoss().to(device),
                 'cell': nn.CrossEntropyLoss().to(device)}

    # mean and std of PubMed Central table images
    normalize = transforms.Normalize(mean=[0.94247851, 0.94254675, 0.94292611],
                                     std=[0.17910956, 0.17940403, 0.17931663])
    mode = 'tag'
    if args.predict_content:
        mode += '+cell'
    if args.predict_bbox:
        mode += '+bbox'
    train_loader = TagCellDataset(args.data_folder, args.data_name, 'TRAIN',
                                  batch_size=args.batch_size, mode=mode,
                                  transform=transforms.Compose([normalize]))

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        decoder.train_epoch(train_loader=train_loader,
                            encoder=encoder,
                            criterion=criterion,
                            encoder_optimizer=encoder_optimizer,
                            tag_decoder_optimizer=tag_decoder_optimizer,
                            cell_decoder_optimizer=cell_decoder_optimizer,
                            cell_bbox_regressor_optimizer=cell_decoder_optimizer,
                            epoch=epoch,
                            args=args)

        # Save checkpoint
        save_checkpoint_dual(args.out_dir, args.data_name, epoch, encoder, decoder, encoder_optimizer,
                             tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer)
