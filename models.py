'''
Implementation of encoder-dual-decoder model
'''
import torch
from torch import nn
import torchvision
from torchvision.models.resnet import BasicBlock, conv1x1
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def resnet_block(stride=1):
    layers = []
    downsample = nn.Sequential(
        conv1x1(256, 512, stride),
        nn.BatchNorm2d(512),
    )
    layers.append(BasicBlock(256, 512, stride, downsample))
    layers.append(BasicBlock(512, 512, 1))
    return nn.Sequential(*layers)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14, use_RNN=False, rnn_size=512, last_layer_stride=2):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.use_RNN = use_RNN
        self.rnn_size = rnn_size

        resnet = torchvision.models.resnet18(pretrained=False)  # ImageNet ResNet-18

        # Remove linear and pool layers (since we're not doing classification)
        # Also remove the last CNN layer for higher resolution feature map
        modules = list(resnet.children())[:-3]
        if last_layer_stride is not None:
            modules.append(resnet_block(stride=last_layer_stride))

        # Change stride of max pooling layer for higher resolution feature map
        # modules[3] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)

        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_image_size, self.enc_image_size))

        if self.use_RNN:
            self.RNN = nn.LSTM(512, self.rnn_size, bias=True, batch_first=True)  # LSTM that transforms the image features
            self.init_h = nn.Linear(512, self.rnn_size)  # linear layer to find initial hidden state of LSTM
            self.init_c = nn.Linear(512, self.rnn_size)  # linear layer to find initial cell state of LSTM
        self.fine_tune()

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out).unsqueeze(0)  # (batch_size*encoded_image_size, rnn_size)
        c = self.init_c(mean_encoder_out).unsqueeze(0)
        return h, c

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        batch_size = images.size(0)
        out = self.resnet(images)  # (batch_size, 512, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 512, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)
        if self.use_RNN:
            out = out.contiguous().view(-1, self.enc_image_size, 512)  # (batch_size*encoded_image_size, encoded_image_size, 512)
            h = self.init_hidden_state(out)
            out, h = self.RNN(out, h)  # (batch_size*encoded_image_size, encoded_image_size, 512)
            out = out.view(batch_size, self.enc_image_size, self.enc_image_size, self.rnn_size).contiguous()
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class CellAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, tag_decoder_dim, language_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param tag_decoder_dim: size of tag decoder's RNN
        :param language_dim: size of language model's RNN
        :param attention_dim: size of the attention network
        """
        super(CellAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.tag_decoder_att = nn.Linear(tag_decoder_dim, attention_dim)  # linear layer to transform tag decoder output
        self.language_att = nn.Linear(language_dim, attention_dim)  # linear layer to transform language models output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden, language_out):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (1, num_pixels, encoder_dim)
        :param decoder_hidden: tag decoder output, a tensor of dimension [(num_cells, tag_decoder_dim)]
        :param language_out: language model output, a tensor of dimension (num_cells, language_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (1, num_pixels, attention_dim)
        att2 = self.tag_decoder_att(decoder_hidden)  # (num_cells, tag_decoder_dim)
        att3 = self.language_att(language_out)  # (num_cells, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))).squeeze(2)  # (num_cells, num_pixels)
        alpha = self.softmax(att)  # (num_cells, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (num_cells, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, decoder_cell=nn.LSTMCell, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        assert decoder_cell.__name__ in ('GRUCell', 'LSTMCell'), 'decoder_cell must be either nn.LSTMCell or nn.GRUCell'
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = decoder_cell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        if isinstance(self.decode_step, nn.LSTMCell):
            self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        if isinstance(self.decode_step, nn.LSTMCell):
            c = self.init_c(mean_encoder_out)
            return h, c
        else:
            return h

    def inference(self, encoder_out, word_map, max_steps=400, beam_size=5, return_attention=False):
        """
        Inference on test images with beam search
        """
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        k = beam_size
        vocab_size = len(word_map)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        if return_attention:
            seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        if return_attention:
            complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1

        if isinstance(self.decode_step, nn.LSTMCell):
            h, c = self.init_hidden_state(encoder_out)
        else:
            h = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            if return_attention:
                awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
            else:
                awe, _ = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            if isinstance(self.decode_step, nn.LSTMCell):
                h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            else:
                h = self.decode_step(torch.cat([embeddings, awe], dim=1), h)  # (s, decoder_dim)

            h = repackage_hidden(h)
            if isinstance(self.decode_step, nn.LSTMCell):
                c = repackage_hidden(c)

            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            if return_attention:
                seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                       dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = []
            complete_inds = []
            for ind, next_word in enumerate(next_word_inds):
                if next_word == word_map['<end>']:
                    complete_inds.append(ind)
                else:
                    incomplete_inds.append(ind)

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                if return_attention:
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break

            # Break if things have been going on too long
            if step > max_steps:
                # If no complete sequence is generated, finish the incomplete
                # sequences with <end>
                if not complete_seqs_scores:
                    complete_seqs = seqs.tolist()
                    for i in range(len(complete_seqs)):
                        complete_seqs[i].append(word_map['<end>'])
                    if return_attention:
                        complete_seqs_alpha = seqs_alpha.tolist()
                    complete_seqs_scores = top_k_scores.tolist()
                break

            seqs = seqs[incomplete_inds]
            if return_attention:
                seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            if isinstance(self.decode_step, nn.LSTMCell):
                c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        if return_attention:
            alphas = complete_seqs_alpha[i]
            return seq, alphas
        else:
            return seq

    def forward(self, encoder_out, encoded_captions, caption_lengths, h, c=None, begin_tokens=None):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        num_pixels = encoder_out.size(1)

        if begin_tokens is None:
            # Embedding
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
            # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
            # So, decoding lengths are actual lengths - 1
            decode_lengths = (caption_lengths - 1).tolist()
        else:  # For TBPTT, use the end token of the previous sub-sequence as begin token instead of <start>
            embeddings = torch.cat([self.embedding(begin_tokens), self.embedding(encoded_captions)], dim=1)
            decode_lengths = caption_lengths.tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, decode_lengths[0], vocab_size).to(device)
        alphas = torch.zeros(batch_size, decode_lengths[0], num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(decode_lengths[0]):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            if isinstance(self.decode_step, nn.LSTMCell):
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    h[:batch_size_t])  # (batch_size_t, decoder_dim)
            predictions[:batch_size_t, t, :] = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            alphas[:batch_size_t, t, :] = alpha

        return predictions, decode_lengths, alphas, h, c

    def train_epoch(self, train_loader, encoder, criterion, encoder_optimizer, decoder_optimizer, epoch, args, step=None):
        """
        Performs one epoch's training.

        :param train_loader: DataLoader for training data
        :param encoder: encoder model
        :param decoder: decoder model
        :param criterion: loss layer
        :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
        :param decoder_optimizer: optimizer to update decoder's weights
        :param epoch: epoch number
        """

        self.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top1accs = AverageMeter()  # top1 accuracy

        start = time.time()

        # Batches
        train_loader.shuffle()
        for i, (imgs, caps_sorted, caplens) in enumerate(train_loader):
            if step is not None:
                if i <= step:
                    continue
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps_sorted = caps_sorted.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            imgs = encoder(imgs)
            # Flatten image
            batch_size = imgs.size(0)
            encoder_dim = imgs.size(-1)
            imgs = imgs.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
            caplens = caplens.squeeze(1)

            # Sort input data by decreasing lengths
            # caplens, sort_ind = caplens.squeeze(1).sort(dim=0, descending=True)
            # imgs = imgs[sort_ind]
            # caps_sorted = caps[sort_ind]

            # Initialize LSTM state
            if isinstance(self.decode_step, nn.LSTMCell):
                h, c = self.init_hidden_state(imgs)  # (batch_size, decoder_dim)
            else:
                h = self.init_hidden_state(imgs)  # (batch_size, decoder_dim)
                c = None

            max_cap_length = max(caplens.tolist())
            # TBPTT
            j = 0
            while j < max_cap_length:
                if j == 0:
                    # bptt tokens after <start>
                    sub_seq_len = min(args.bptt + 1, max_cap_length - j)
                else:
                    sub_seq_len = min(args.bptt, max_cap_length - j)
                    # Do not leave too short tails (less than 10 tokens)
                    short_tail = (caplens - (j + sub_seq_len) < 10) & (caplens - (j + sub_seq_len) > 0)
                    if short_tail.any():
                        sub_seq_len += max((caplens - (j + sub_seq_len))[short_tail].tolist())

                sub_seq_caplens = caplens - j
                sub_seq_caplens[sub_seq_caplens > sub_seq_len] = sub_seq_len
                batch_size_t = (sub_seq_caplens > 0).sum().item()
                sub_seq_caplens = sub_seq_caplens[:batch_size_t]
                sub_seq_cap = caps_sorted[:batch_size_t, j:j + sub_seq_len]

                h = repackage_hidden(h)
                if isinstance(self.decode_step, nn.LSTMCell):
                    c = repackage_hidden(c)

                decoder_optimizer.zero_grad()
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()
                if j == 0:
                    scores, decode_lengths, alphas, h, c = self(
                        imgs[:batch_size_t],
                        sub_seq_cap,
                        sub_seq_caplens,
                        h,
                        c)
                    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                    targets = sub_seq_cap[:, 1:]
                else:
                    scores, decode_lengths, alphas, h, c = self(
                        imgs[:batch_size_t],
                        sub_seq_cap,
                        sub_seq_caplens,
                        h,
                        c,
                        caps_sorted[:batch_size_t, j - 1].unsqueeze(1))
                    targets = sub_seq_cap

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Back prop.
                if j + sub_seq_len < max_cap_length:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                # Clip gradients
                if args.grad_clip is not None:
                    clip_gradient(decoder_optimizer, args.grad_clip)
                    if encoder_optimizer is not None:
                        clip_gradient(encoder_optimizer, args.grad_clip)

                # Update weights
                decoder_optimizer.step()
                if encoder_optimizer is not None:
                    encoder_optimizer.step()

                # Keep track of metrics
                top1 = accuracy(scores, targets, 1)
                losses.update(loss.item(), sum(decode_lengths))
                top1accs.update(top1, sum(decode_lengths))
                j += sub_seq_len
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top1=top1accs), file=sys.stderr)
                sys.stderr.flush()

class DecoderWithAttentionAndLanguageModel(nn.Module):
    '''
    Stacked 2-layer LSTM with Attention model. First LSTM is a languange model, second LSTM is a decoder.
    See "Recursive Recurrent Nets with Attention Modeling for OCR in the Wild"
    '''
    def __init__(self, attention_dim, embed_dim, language_dim, decoder_dim, vocab_size,
                 decoder_cell=nn.LSTMCell, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param language_dim: size of language model's RNN
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttentionAndLanguageModel, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.language_dim = language_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, language_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.decode_step_LM = decoder_cell(embed_dim, language_dim, bias=True)  # language model LSTMCell

        self.decode_step_pred = decoder_cell(encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        if isinstance(self.decode_step_pred, nn.LSTMCell):
            self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        batch_size = encoder_out.size(0)
        mean_encoder_out = encoder_out.mean(dim=1)
        h_LM = torch.zeros(batch_size, self.language_dim).to(device)
        h_pred = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        if isinstance(self.decode_step_pred, nn.LSTMCell):
            c_LM = torch.zeros(batch_size, self.language_dim).to(device)
            c_pred = self.init_c(mean_encoder_out)
            return h_LM, c_LM, h_pred, c_pred
        else:
            return h_LM, h_pred

    def inference(self, encoder_out, word_map, max_steps=400, beam_size=5):
        """
        Inference on test images with beam search
        """
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        k = beam_size
        vocab_size = len(word_map)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        if isinstance(self.decode_step_pred, nn.LSTMCell):
            h_LM, c_LM, h_cell, c_cell = self.init_hidden_state(encoder_out)
        else:
            h_LM, h_cell = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            if isinstance(self.decode_step_LM, nn.LSTMCell):
                h_LM, c_LM = self.decode_step_LM(embeddings, (h_LM, c_LM))
            else:
                h_LM = self.decode_step_LM(embeddings, h_LM)
            awe, alpha = self.attention(encoder_out, h_LM)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = self.sigmoid(self.f_beta(h_cell))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            if isinstance(self.decode_step_pred, nn.LSTMCell):
                h_cell, c_cell = self.decode_step_pred(awe, (h_cell, c_cell))  # (batch_size_t, decoder_dim)
            else:
                h_cell = self.decode_step_pred(awe, h_cell)  # (batch_size_t, decoder_dim)

            scores = self.fc(h_cell)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                   dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = []
            complete_inds = []
            for ind, next_word in enumerate(next_word_inds):
                if next_word == word_map['<end>']:
                    complete_inds.append(ind)
                else:
                    incomplete_inds.append(ind)

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h_LM = h_LM[prev_word_inds[incomplete_inds]]
            h_cell = h_cell[prev_word_inds[incomplete_inds]]
            if isinstance(self.decode_step_pred, nn.LSTMCell):
                c_LM = c_LM[prev_word_inds[incomplete_inds]]
                c_cell = c_cell[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > max_steps:
                break
            step += 1
        if complete_seqs_scores:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            alphas = complete_seqs_alpha[i]
            return seq, alphas
        else:
            return None

    def forward(self, encoder_out, encoded_captions, caption_lengths, h_LM, h_pred, c_LM=None, c_pred=None, begin_tokens=None):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        num_pixels = encoder_out.size(1)

        if begin_tokens is None:
            # Embedding
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
            # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
            # So, decoding lengths are actual lengths - 1
            decode_lengths = (caption_lengths - 1).tolist()
        else:  # For TBPTT, use the end token of the previous sub-sequence as begin token instead of <start>
            embeddings = torch.cat([self.embedding(begin_tokens), self.embedding(encoded_captions)], dim=1)
            decode_lengths = caption_lengths.tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            # Language LSTM
            if isinstance(self.decode_step_LM, nn.LSTMCell):
                h_LM, c_LM = self.decode_step_LM(
                    embeddings[:batch_size_t, t, :],
                    (h_LM[:batch_size_t], c_LM[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h_LM = self.decode_step_LM(
                    embeddings[:batch_size_t, t, :],
                    h_LM[:batch_size_t])  # (batch_size_t, decoder_dim)

            # Attention
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h_LM)

            # Decoder LSTM
            gate = self.sigmoid(self.f_beta(h_pred[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            if isinstance(self.decode_step_pred, nn.LSTMCell):
                h_pred, c_pred = self.decode_step_pred(
                    attention_weighted_encoding,
                    (h_pred[:batch_size_t], c_pred[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h_pred = self.decode_step_pred(
                    attention_weighted_encoding,
                    h_pred[:batch_size_t])  # (batch_size_t, decoder_dim)
            predictions[:batch_size_t, t, :] = self.fc(self.dropout(h_pred))  # (batch_size_t, vocab_size)
            alphas[:batch_size_t, t, :] = alpha

        return predictions, decode_lengths, alphas, h_LM, h_pred, c_LM, c_pred

    def train_epoch(self, train_loader, encoder, criterion, encoder_optimizer, decoder_optimizer, epoch, args, step=None):
        """
        Performs one epoch's training.

        :param train_loader: DataLoader for training data
        :param encoder: encoder model
        :param decoder: decoder model
        :param criterion: loss layer
        :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
        :param decoder_optimizer: optimizer to update decoder's weights
        :param epoch: epoch number
        """

        self.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top1accs = AverageMeter()  # top1 accuracy

        start = time.time()

        # Batches
        train_loader.shuffle()
        for i, (imgs, caps_sorted, caplens) in enumerate(train_loader):
            if step is not None:
                if i <= step:
                    continue
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps_sorted = caps_sorted.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            imgs = encoder(imgs)
            # Flatten image
            batch_size = imgs.size(0)
            encoder_dim = imgs.size(-1)
            imgs = imgs.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
            caplens = caplens.squeeze(1)

            # Sort input data by decreasing lengths
            # caplens, sort_ind = caplens.squeeze(1).sort(dim=0, descending=True)
            # imgs = imgs[sort_ind]
            # caps_sorted = caps[sort_ind]

            # Initialize LSTM state
            if isinstance(self.decode_step_pred, nn.LSTMCell):
                h_LM, c_LM, h_pred, c_pred = self.init_hidden_state(imgs)  # (batch_size, decoder_dim)
            else:
                h_LM, h_pred = self.init_hidden_state(imgs)  # (batch_size, decoder_dim)
                c_LM = c_pred = None

            max_cap_length = max(caplens.tolist())
            # TBPTT
            j = 0
            while j < max_cap_length:
                if j == 0:
                    # bptt tokens after <start>
                    sub_seq_len = min(args.bptt + 1, max_cap_length - j)
                else:
                    sub_seq_len = min(args.bptt, max_cap_length - j)
                    # Do not leave too short tails (less than 10 tokens)
                    short_tail = (caplens - (j + sub_seq_len) < 10) & (caplens - (j + sub_seq_len) > 0)
                    if short_tail.any():
                        sub_seq_len += max((caplens - (j + sub_seq_len))[short_tail].tolist())

                sub_seq_caplens = caplens - j
                sub_seq_caplens[sub_seq_caplens > sub_seq_len] = sub_seq_len
                batch_size_t = (sub_seq_caplens > 0).sum().item()
                sub_seq_caplens = sub_seq_caplens[:batch_size_t]
                sub_seq_cap = caps_sorted[:batch_size_t, j:j + sub_seq_len]

                h_LM = repackage_hidden(h_LM)
                h_pred = repackage_hidden(h_pred)
                if isinstance(self.decode_step_pred, nn.LSTMCell):
                    c_LM = repackage_hidden(c_LM)
                    c_pred = repackage_hidden(c_pred)

                decoder_optimizer.zero_grad()
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()
                if j == 0:
                    scores, decode_lengths, alphas, h_LM, h_pred, c_LM, c_pred = self(
                        imgs[:batch_size_t],
                        sub_seq_cap,
                        sub_seq_caplens,
                        h_LM, h_pred, c_LM, c_pred)
                    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                    targets = sub_seq_cap[:, 1:]
                else:
                    scores, decode_lengths, alphas, h_LM, h_pred, c_LM, c_pred = self(
                        imgs[:batch_size_t],
                        sub_seq_cap,
                        sub_seq_caplens,
                        h_LM, h_pred, c_LM, c_pred,
                        caps_sorted[:batch_size_t, j - 1].unsqueeze(1))
                    targets = sub_seq_cap

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Back prop.
                if j + sub_seq_len < max_cap_length:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                # Clip gradients
                if args.grad_clip is not None:
                    clip_gradient(decoder_optimizer, args.grad_clip)
                    if encoder_optimizer is not None:
                        clip_gradient(encoder_optimizer, args.grad_clip)

                # Update weights
                decoder_optimizer.step()
                if encoder_optimizer is not None:
                    encoder_optimizer.step()

                # Keep track of metrics
                top1 = accuracy(scores, targets, 1)
                losses.update(loss.item(), sum(decode_lengths))
                top1accs.update(top1, sum(decode_lengths))
                j += sub_seq_len
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top1=top1accs), file=sys.stderr)
                sys.stderr.flush()

class TagDecoder(DecoderWithAttention):
    '''
    TagDecoder generates structure of the table
    '''
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 td_encode, decoder_cell=nn.LSTMCell, encoder_dim=512,
                 dropout=0.5, cnn_layer_stride=None, tag_H_grad=True):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(TagDecoder, self).__init__(
            attention_dim,
            embed_dim,
            decoder_dim,
            vocab_size,
            decoder_cell,
            encoder_dim,
            dropout)
        self.td_encode = td_encode
        self.tag_H_grad = tag_H_grad
        if cnn_layer_stride is not None:
            self.input_filter = resnet_block(cnn_layer_stride)

    def inference(self, encoder_out, word_map, max_steps=400, beam_size=5, return_attention=False):
        """
        Inference on test images with beam search
        """
        if hasattr(self, 'input_filter'):
            encoder_out = self.input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        k = beam_size
        vocab_size = len(word_map)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        if return_attention:
            seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        if return_attention:
            complete_seqs_alpha = list()
        complete_seqs_scores = list()
        complete_seqs_tag_H = list()

        # Start decoding
        step = 1
        if isinstance(self.decode_step, nn.LSTMCell):
            h, c = self.init_hidden_state(encoder_out)
        else:
            h = self.init_hidden_state(encoder_out)
        tag_H = [[] for i in range(k)]
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            if return_attention:
                awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
            else:
                awe, _ = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            if isinstance(self.decode_step, nn.LSTMCell):
                h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            else:
                h = self.decode_step(torch.cat([embeddings, awe], dim=1), h)  # (s, decoder_dim)

            h = repackage_hidden(h)
            if isinstance(self.decode_step, nn.LSTMCell):
                c = repackage_hidden(c)

            for i, w in enumerate(k_prev_words):
                if w[0].item() in (word_map['<td>'], word_map['>']):
                    tag_H[i].append(h[i].unsqueeze(0))

            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)


            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            if return_attention:
                seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                       dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = []
            complete_inds = []
            for ind, next_word in enumerate(next_word_inds):
                if next_word == word_map['<end>']:
                    complete_inds.append(ind)
                else:
                    incomplete_inds.append(ind)

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                if return_attention:
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_seqs_tag_H.extend([tag_H[i].copy() for i in prev_word_inds[complete_inds]])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Break if all sequences are complete
            if k == 0:
                break
            # Break if things have been going on too long
            if step > max_steps:
                # If no complete sequence is generated, finish the incomplete
                # sequences with <end>
                if not complete_seqs_scores:
                    complete_seqs = seqs.tolist()
                    for i in range(len(complete_seqs)):
                        complete_seqs[i].append(word_map['<end>'])
                    if return_attention:
                        complete_seqs_alpha = seqs_alpha.tolist()
                    complete_seqs_scores = top_k_scores.tolist()
                    complete_seqs_tag_H = [tag_H[i].copy() for i in prev_word_inds]
                break

            # Proceed with incomplete sequences
            seqs = seqs[incomplete_inds]
            if return_attention:
                seqs_alpha = seqs_alpha[incomplete_inds]
            tag_H = [tag_H[i].copy() for i in prev_word_inds[incomplete_inds]]
            h = h[prev_word_inds[incomplete_inds]]
            if isinstance(self.decode_step, nn.LSTMCell):
                c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        if complete_seqs_tag_H[i]:
            tag_H = torch.cat(complete_seqs_tag_H[i]).to(device)
        else:
            tag_H = torch.zeros(0).to(device)
        if return_attention:
            alphas = complete_seqs_alpha[i]
            return seq, alphas, tag_H
        else:
            return seq, tag_H

    def forward(self, encoder_out, encoded_tags_sorted, tag_lengths, num_cells=None, max_tag_len=None):
        # Flatten image
        if hasattr(self, 'input_filter'):
            encoder_out = self.input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embedding(encoded_tags_sorted)  # (batch_size, max_caption_length, embed_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (tag_lengths - 1).tolist()
        max_decode_lengths = decode_lengths[0] if max_tag_len is None else max_tag_len
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max_decode_lengths, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_decode_lengths, num_pixels).to(device)

        if num_cells is not None:
            # Create tensors to hold hidden state of tag decoder for cell decoder
            tag_H = [torch.zeros(n.item(), self.decoder_dim).to(device) for n in num_cells]
            pointer = torch.zeros(batch_size, dtype=torch.long).to(device)

        # Initialize LSTM state
        if isinstance(self.decode_step, nn.LSTMCell):
            h, c = self.init_hidden_state(encoder_out)
        else:
            h = self.init_hidden_state(encoder_out)

        # Decode table structure
        for t in range(max_decode_lengths):
            batch_size_t = sum([l > t for l in decode_lengths])
            if batch_size_t > 0:
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                if isinstance(self.decode_step, nn.LSTMCell):
                    h, c = self.decode_step(
                        torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                        (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                else:
                    h = self.decode_step(
                        torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                        h[:batch_size_t])  # (batch_size_t, decoder_dim)
                predictions[:batch_size_t, t, :] = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
                alphas[:batch_size_t, t, :] = alpha
                if num_cells is not None:
                    for i in range(batch_size_t):
                        if encoded_tags_sorted[i, t] in self.td_encode:
                            if self.tag_H_grad:
                                tag_H[i][pointer[i]] = h[i]
                            else:
                                tag_H[i][pointer[i]] = repackage_hidden(h[i])
                            pointer[i] += 1
        if num_cells is None:
            return predictions, decode_lengths, alphas
        else:
            return predictions, decode_lengths, alphas, tag_H

    def train_epoch(self, train_loader, encoder, criterion, encoder_optimizer, decoder_optimizer, epoch, args):
        """
        Performs one epoch's training.

        :param train_loader: DataLoader for training data
        :param encoder: encoder model
        :param criterion: loss layer
        :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
        :param decoder_optimizer: optimizer to update decoder's weights
        :param epoch: epoch number
        """
        self.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        losses = AverageMeter()  # loss (per word decoded)
        top1accs = AverageMeter()  # top1 accuracy

        start = time.time()
        # Batches
        for i, (imgs, tags, tag_lens) in enumerate(train_loader):
            # Move to GPU, if available
            imgs = imgs.to(device)
            tags = tags.to(device)
            tag_lens = tag_lens.to(device)

            # Flatten image
            batch_size = imgs.size(0)
            encoder_dim = imgs.size(-1)
            imgs = imgs.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
            tag_lens = tag_lens.squeeze(1)

            # Sort input data by decreasing lengths
            tag_lens, sort_ind = tag_lens.sort(dim=0, descending=True)
            imgs = imgs[sort_ind]
            tags_sorted = tags[sort_ind]

            # Forward prop.
            imgs = encoder(imgs)
            if hasattr(self, 'input_filter'):
                imgs = self.input_filter(imgs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            scores_tag, decode_lengths_tag, alphas_tag = self(
                imgs, tags_sorted, tag_lens)

            # Calculate tag loss
            targets_tag = tags_sorted[:, 1:]
            scores_tag = pack_padded_sequence(scores_tag, decode_lengths_tag, batch_first=True)[0]
            targets_tag = pack_padded_sequence(targets_tag, decode_lengths_tag, batch_first=True)[0]
            loss = criterion(scores_tag, targets_tag)
            # Add doubly stochastic attention regularization
            loss += args.alpha_c * ((1. - alphas_tag.sum(dim=1)) ** 2).mean()
            top1 = accuracy(scores_tag, targets_tag, 1)
            tag_count = sum(decode_lengths_tag)
            losses.update(loss.item(), tag_count)
            top1accs.update(top1, tag_count)

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if args.grad_clip is not None:
                clip_gradient(decoder_optimizer, args.grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, args.grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            batch_time.update(time.time() - start)
            start = time.time()

            # Print status
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                        batch_time=batch_time,
                                                                        loss=losses,
                                                                        top1=top1accs), file=sys.stderr)
                sys.stderr.flush()

class BBoxLoss(nn.Module):
    def __init__(self):
        super(BBoxLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()

    def bbox_loss(self, gt, pred):
        center_loss = (gt[:, :2] - pred[:, :2]).square().sum(dim=1)
        size_loss = (gt[:, 2:].sqrt() - pred[:, 2:].sqrt()).square().sum(dim=1)
        return center_loss + size_loss

    def forward(self, gt, pred):
        empty_loss = self.CE(pred[:, :2], gt[:, 0].long())  # Empty cell classification loss
        bbox_loss = (gt[:, 0] * self.bbox_loss(gt[:, 1:], pred[:, 2:])).mean()   # Only compute for non-empty cells
        return empty_loss + bbox_loss

class CellBBox(nn.Module):
    """
    Regression network for bbox of table cell.
    """

    def __init__(self, tag_decoder_dim):
        """
        :param tag_decoder_dim: size of tag decoder's RNN
        """
        super(CellBBox, self).__init__()
        # linear layers to predict bbox (x_c, y_c, w, h)
        self.bbox = nn.Sequential(
            nn.Linear(tag_decoder_dim, tag_decoder_dim),
            nn.ReLU(),
            nn.Linear(tag_decoder_dim, 4),
            nn.Sigmoid()
        )
        # linear layers to predict if a cell is empty
        self.empty_cls = nn.Sequential(
            nn.Linear(tag_decoder_dim, tag_decoder_dim),
            nn.ReLU(),
            nn.Linear(tag_decoder_dim, 2)
        )

    def forward(self, decoder_hidden):
        """
        Forward propagation.
        :param decoder_hidden: tag decoder output, a tensor of dimension (batch_size, tag_decoder_dim)
        """
        batch_size = encoder_out.size(0)
        output = []
        for i in range(batch_size):
            not_empty = self.empty_cls(decoder_hidden[i])  # (num_cells, 2)
            bbox_pred = self.bbox(decoder_hidden[i])  # (num_cells, 4)
            output.append(torch.cat([not_empty, bbox_pred]))  # (num_cells, 6)
        return output


class BBoxLoss_Yolo(nn.Module):
    def __init__(self, w_coor=5.0, w_noobj=0.5, image_size=(28, 28)):
        super(BBoxLoss_Yolo, self).__init__()
        self.w_coor = w_coor
        self.w_noobj = w_noobj
        self.image_size = image_size

    def IoU(self, pred, idx, gt):
        ''' Calculates IoU between prediction boxes and table cell
        '''
        pred_xmin = pred[:, 1::5] - pred[:, 3::5] / 2
        pred_xmax = pred[:, 1::5] + pred[:, 3::5] / 2
        pred_ymin = pred[:, 2::5] - pred[:, 4::5] / 2
        pred_ymax = pred[:, 2::5] + pred[:, 4::5] / 2
        gt_xmin = (gt[:, 1] - gt[:, 3] / 2).unsqueeze(1)
        gt_xmax = (gt[:, 1] + gt[:, 3] / 2).unsqueeze(1)
        gt_ymin = (gt[:, 2] - gt[:, 4] / 2).unsqueeze(1)
        gt_ymax = (gt[:, 2] + gt[:, 4] / 2).unsqueeze(1)

        I_w = torch.max(torch.FloatTensor([0]), torch.min(pred_xmax, gt_xmax) - torch.max(pred_xmin, gt_xmin))
        I_h = torch.max(torch.FloatTensor([0]), torch.min(pred_ymax, gt_ymax) - torch.max(pred_ymin, gt_ymin))
        I = I_w * I_h
        U = pred[:, 3::5] * pred[:, 4::5] + (gt[:, 3] * gt[:, 4]).unsqueeze(1) - I
        IoU = I / (U + 1e-8)  # Avoid dividing by 0
        return IoU

    def find_responsible_box(self, pred, idx, gt):
        ''' Finds which prediction box is responsible for the table cell
        '''
        pred = pred[idx[0], idx[1]]
        IoU = self.IoU(pred, gt)
        num_cells = gt.size(0)
        IoU, responsible_box = torch.max(IoU, dim=1)
        return responsible_box, IoU

    def forward(self, gt, pred):
        '''
        :param gt: ground truth (num_cells, 5)
        :param pred: prediction of CellBBoxYolo (num_cells, num_pixels, 5 * num_bboxes_per_pixel)
        '''
        num_cells = gt.size(0)
        image_width, image_height = self.image_size28
        non_empty_cell = gt[:, 0] == 1

        gt_non_empty, gt_empty = gt[non_empty_cell], gt[~non_empty_cell]
        pred_non_empty, pred_empty = pred[non_empty_cell], pred[~non_empty_cell]
        loss_empty = self.w_noobj * pred_empty[:, :, 0::5].square().sum()

        # Encode gt as Yolo format
        # Find center pixel
        x_c, y_c = torch.floor(gt_non_empty[:, 1] * image_width), torch.floor(gt_non_empty[:, 2] * image_height)
        idx = (torch.LongTensor(torch.arange(gt_non_empty.size(0))), (x_c * image_width + y_c).long())

        # Compute offset
        gt_non_empty[:, 1], gt_non_empty[:, 2] = gt_non_empty[:, 1] * image_width - x_c, gt_non_empty[:, 2] * image_height - y_c
        gt_non_empty[:, 3], gt_non_empty[:, 4] = gt_non_empty[:, 3] * image_width, gt_non_empty[:, 4] * image_height

        responsible_box, IoU = self.find_responsible_box(pred_non_empty, idx, gt_non_empty)
        responsible_box = responsible_box * 5
        gt_non_empty[:, 0] = IoU
        gt_non_empty[:, 3:5] = gt_non_empty[:, 3:5].sqrt()

        responsible_box = torch.cat((
            pred_non_empty[idx[0], idx[1], responsible_box],
            pred_non_empty[idx[0], idx[1], responsible_box + 1],
            pred_non_empty[idx[0], idx[1], responsible_box + 2],
            pred_non_empty[idx[0], idx[1], responsible_box + 3].sqrt(),
            pred_non_empty[idx[0], idx[1], responsible_box + 4].sqrt()
        ), dim=1)


        loss_coor = self.w_coor * (responsible_box[:, 1:5] - gt_non_empty[:, 1:5]).square().sum()
        loss_noobj = (responsible_box[:, 0] - gt_non_empty[:, 0]).square().sum() + \
                     self.w_noobj * 0 + \
                     loss_empty

        return loss_coor + loss_noobj


class CellBBoxYolo(nn.Module):
    """
    NOT READY
    Table cell detection network (based on the idea of Yolo).
    """

    def __init__(self, encoder_dim, tag_decoder_dim, feature_dim, num_bboxes_per_pixel=2):
        """
        :param encoder_dim: feature size of encoded images
        :param tag_decoder_dim: size of tag decoder's RNN
        :param feature_dim: size of the features
        """
        super(CellBBoxYolo, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, feature_dim)  # linear layer to transform encoded image
        self.tag_decoder_att = nn.Linear(tag_decoder_dim, feature_dim)  # linear layer to transform tag decoder output
        self.bbox = nn.Linear(feature_dim, 5 * num_bboxes_per_pixel)  # linear layer to predict bboxes [c, x_c, y_c, w, h] * num_bboxes_per_pixel
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # sigmoid to scale bbox between 0 and 1

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: tag decoder output, a tensor of dimension [(num_cells, tag_decoder_dim)] * batch_size
        :return: [(num_cells,  5 * num_bboxes_per_pixel)] * batch_size
        """
        batch_size = encoder_out.size(0)
        output = []
        for i in range(batch_size):
            att1 = self.encoder_att(encoder_out[i].unsqueeze(0))  # (1, num_pixels, feature_dim)
            att2 = self.tag_decoder_att(decoder_hidden[i]).unsqueeze(1)  # (num_cells, 1, feature_dim)
            att = self.relu(att1 + att2)  # (num_cells, num_pixels, feature_dim)
            bboxes = self.sigmoid(self.bbox(att))  # (num_cells, num_pixels, 5 * num_bboxes_per_pixel)
            output.append(bboxes)
        return output


class CellDecoder_baseline(nn.Module):
    '''
    CellDecoder generates cell content
    '''
    def __init__(self, attention_dim, embed_dim, tag_decoder_dim, decoder_dim,
                 vocab_size, decoder_cell=nn.LSTMCell, encoder_dim=512,
                 dropout=0.5, cnn_layer_stride=None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param tag_decoder_dim: size of tag decoder's RNN
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param mini_batch_size: batch size of cells to reduce GPU memory usage
        """
        super(CellDecoder_baseline, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = CellAttention(encoder_dim, tag_decoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.decode_step = decoder_cell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoder LSTMCell

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        if isinstance(self.decode_step, nn.LSTMCell):
            self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.dropout = nn.Dropout(p=self.dropout)

        if cnn_layer_stride is not None:
            self.input_filter = resnet_block(cnn_layer_stride)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out).expand(batch_size, -1)
        if isinstance(self.decode_step, nn.LSTMCell):
            c = self.init_c(mean_encoder_out).expand(batch_size, -1)
            return h, c
        else:
            return h

    def inference(self, encoder_out, tag_H, word_map, max_steps=400, beam_size=5, return_attention=False):
        """
        Inference on test images with beam search
        """

        if hasattr(self, 'input_filter'):
            encoder_out = self.input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        num_cells = tag_H.size(0)
        cell_seqs = []
        if return_attention:
            cell_alphas = []
        vocab_size = len(word_map)

        for c_id in range(num_cells):
            k = beam_size
            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            if return_attention:
                # Tensor to store top k sequences' alphas; now they're just 1s
                seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

            # Lists to store completed sequences, their alphas and scores
            complete_seqs = list()
            if return_attention:
                complete_seqs_alpha = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            if isinstance(self.decode_step, nn.LSTMCell):
                h, c = self.init_hidden_state(encoder_out, k)
            else:
                h = self.init_hidden_state(encoder_out, k)

            cell_tag_H = tag_H[c_id].expand(k, -1)
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                if return_attention:
                    awe, alpha = self.attention(encoder_out, cell_tag_H, h)  # (s, encoder_dim), (s, num_pixels)
                    alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
                else:
                    awe, _ = self.attention(encoder_out, cell_tag_H, h)  # (s, encoder_dim), (s, num_pixels)

                gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                if isinstance(self.decode_step, nn.LSTMCell):
                    h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                else:
                    h = self.decode_step(torch.cat([embeddings, awe], dim=1), h)  # (s, decoder_dim)

                h = repackage_hidden(h)
                if isinstance(self.decode_step, nn.LSTMCell):
                    c = repackage_hidden(c)

                scores = self.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences, alphas
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                if return_attention:
                    seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                           dim=1)  # (s, step+1, enc_image_size, enc_image_size)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = []
                complete_inds = []
                for ind, next_word in enumerate(next_word_inds):
                    if next_word == word_map['<end>']:
                        complete_inds.append(ind)
                    else:
                        incomplete_inds.append(ind)

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    if return_attention:
                        complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Break if all sequences are complete
                if k == 0:
                    break
                # Break if things have been going on too long
                if step > max_steps:
                    # If no complete sequence is generated, finish the incomplete
                    # sequences with <end>
                    if not complete_seqs_scores:
                        complete_seqs = seqs.tolist()
                        for i in range(len(complete_seqs)):
                            complete_seqs[i].append(word_map['<end>'])
                        if return_attention:
                            complete_seqs_alpha = seqs_alpha.tolist()
                        complete_seqs_scores = top_k_scores.tolist()
                    break

                # Proceed with incomplete sequences
                seqs = seqs[incomplete_inds]
                if return_attention:
                    seqs_alpha = seqs_alpha[incomplete_inds]
                cell_tag_H = cell_tag_H[prev_word_inds[incomplete_inds]]
                h = h[prev_word_inds[incomplete_inds]]
                if isinstance(self.decode_step, nn.LSTMCell):
                    c = c[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                step += 1
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            cell_seqs.append(complete_seqs[i])
            if return_attention:
                cell_alphas.append(complete_seqs_alpha[i])
        if return_attention:
            return cell_seqs, cell_alphas
        else:
            return cell_seqs

    def forward(self, encoder_out, encoded_cells_sorted, cell_lengths, tag_H):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_cells_sorted: encoded cells, a list of batch_size tensors of dimension (num_cells, max_cell_length)
        :param tag_H: hidden state from TagDeoder, a list of batch_size tensors of dimension (num_cells, TagDecoder's decoder_dim)
        :param cell_lengths: caption lengths, a list of batch_size tensor of dimension (num_cells, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights
        """
        if hasattr(self, 'input_filter'):
            encoder_out = self.input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Flatten image
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Decode cell content
        predictions_cell = []
        alphas_cell = []
        decode_lengths_cell = []
        for i in range(batch_size):
            num_cells = cell_lengths[i].size(0)
            embeddings = self.embedding(encoded_cells_sorted[i])
            decode_lengths = (cell_lengths[i] - 1).tolist()
            max_decode_lengths = decode_lengths[0]
            predictions = torch.zeros(num_cells, max_decode_lengths, self.vocab_size).to(device)
            alphas = torch.zeros(num_cells, max_decode_lengths, num_pixels).to(device)
            if isinstance(self.decode_step, nn.LSTMCell):
                h, c = self.init_hidden_state(encoder_out[i].unsqueeze(0), num_cells)
            else:
                h = self.init_hidden_state(encoder_out[i].unsqueeze(0), num_cells)
            for t in range(max_decode_lengths):
                batch_size_t = sum([l > t for l in decode_lengths])
                attention_weighted_encoding, alpha = self.attention(encoder_out[i].unsqueeze(0),
                                                                    tag_H[i][:batch_size_t],
                                                                    h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                if isinstance(self.decode_step, nn.LSTMCell):
                    h, c = self.decode_step(
                        torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                        (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                else:
                    h = self.decode_step(
                        torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                        h[:batch_size_t])  # (batch_size_t, decoder_dim)
                predictions[:batch_size_t, t, :] = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
                alphas[:batch_size_t, t, :] = alpha
            predictions_cell.append(predictions)
            alphas_cell.append(alphas)
            decode_lengths_cell.append(decode_lengths)
        return predictions_cell, decode_lengths_cell, alphas_cell

class CellDecoder(nn.Module):
    '''
    CellDecoder generates cell content
    '''
    def __init__(self, attention_dim, embed_dim, tag_decoder_dim, language_dim,
                 decoder_dim, vocab_size, decoder_cell=nn.LSTMCell,
                 encoder_dim=512, dropout=0.5, cnn_layer_stride=None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param tag_decoder_dim: size of tag decoder's RNN
        :param language_dim: size of language model's RNN
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param mini_batch_size: batch size of cells to reduce GPU memory usage
        """
        super(CellDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.language_dim = language_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = CellAttention(encoder_dim, tag_decoder_dim, language_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.decode_step_LM = decoder_cell(embed_dim, language_dim, bias=True)  # language model LSTMCell

        self.decode_step_pred = decoder_cell(encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        if isinstance(self.decode_step_pred, nn.LSTMCell):
            self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.dropout = nn.Dropout(p=self.dropout)

        if cnn_layer_stride is not None:
            self.input_filter = resnet_block(cnn_layer_stride)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h_LM = torch.zeros(batch_size, self.language_dim).to(device)
        h_pred = self.init_h(mean_encoder_out).expand(batch_size, -1)
        if isinstance(self.decode_step_pred, nn.LSTMCell):
            c_LM = torch.zeros(batch_size, self.language_dim).to(device)
            c_pred = self.init_c(mean_encoder_out).expand(batch_size, -1)
            return h_LM, c_LM, h_pred, c_pred
        else:
            return h_LM, h_pred

    def inference(self, encoder_out, tag_H, word_map, max_steps=400, beam_size=5, return_attention=False):
        """
        Inference on test images with beam search
        """
        if hasattr(self, 'input_filter'):
            encoder_out = self.input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        num_cells = tag_H.size(0)
        cell_seqs = []
        if return_attention:
            cell_alphas = []
        vocab_size = len(word_map)

        for c in range(num_cells):
            k = beam_size
            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            if return_attention:
                # Tensor to store top k sequences' alphas; now they're just 1s
                seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

            # Lists to store completed sequences, their alphas and scores
            complete_seqs = list()
            if return_attention:
                complete_seqs_alpha = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            if isinstance(self.decode_step_pred, nn.LSTMCell):
                h_LM, c_LM, h_cell, c_cell = self.init_hidden_state(encoder_out, k)
            else:
                h_LM, h_cell = self.init_hidden_state(encoder_out, k)

            cell_tag_H = tag_H[c].expand(k, -1)
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                if isinstance(self.decode_step_LM, nn.LSTMCell):
                    h_LM, c_LM = self.decode_step_LM(embeddings, (h_LM, c_LM))
                else:
                    h_LM = self.decode_step_LM(embeddings, h_LM)

                if return_attention:
                    awe, alpha = self.attention(
                        encoder_out,
                        cell_tag_H,
                        h_LM)
                    alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
                else:
                    awe, _ = self.attention(
                        encoder_out,
                        cell_tag_H,
                        h_LM)
                gate = self.sigmoid(self.f_beta(h_cell))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                if isinstance(self.decode_step_pred, nn.LSTMCell):
                    h_cell, c_cell = self.decode_step_pred(awe, (h_cell, c_cell))  # (batch_size_t, decoder_dim)
                else:
                    h_cell = self.decode_step_pred(awe, h_cell)  # (batch_size_t, decoder_dim)

                h_LM = repackage_hidden(h_LM)
                h_cell = repackage_hidden(h_cell)
                if isinstance(self.decode_step_pred, nn.LSTMCell):
                    c_LM = repackage_hidden(c_LM)
                    c_cell = repackage_hidden(c_cell)

                scores = self.fc(h_cell)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences, alphas
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                if return_attention:
                    seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                           dim=1)  # (s, step+1, enc_image_size, enc_image_size)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = []
                complete_inds = []
                for ind, next_word in enumerate(next_word_inds):
                    if next_word == word_map['<end>']:
                        complete_inds.append(ind)
                    else:
                        incomplete_inds.append(ind)

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    if return_attention:
                        complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Break if all sequences are complete
                if k == 0:
                    break
                # Break if things have been going on too long
                if step > max_steps:
                    # If no complete sequence is generated, finish the incomplete
                    # sequences with <end>
                    if not complete_seqs_scores:
                        complete_seqs = seqs.tolist()
                        for i in range(len(complete_seqs)):
                            complete_seqs[i].append(word_map['<end>'])
                        if return_attention:
                            complete_seqs_alpha = seqs_alpha.tolist()
                        complete_seqs_scores = top_k_scores.tolist()
                    break

                # Proceed with incomplete sequences
                seqs = seqs[incomplete_inds]
                if return_attention:
                    seqs_alpha = seqs_alpha[incomplete_inds]
                cell_tag_H = cell_tag_H[prev_word_inds[incomplete_inds]]
                h_LM = h_LM[prev_word_inds[incomplete_inds]]
                h_cell = h_cell[prev_word_inds[incomplete_inds]]
                if isinstance(self.decode_step_pred, nn.LSTMCell):
                    c_LM = c_LM[prev_word_inds[incomplete_inds]]
                    c_cell = c_cell[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                step += 1
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            cell_seqs.append(complete_seqs[i])
            if return_attention:
                cell_alphas.append(complete_seqs_alpha[i])
        if return_attention:
            return cell_seqs, cell_alphas
        else:
            return cell_seqs

    def forward(self, encoder_out, encoded_cells_sorted, cell_lengths, tag_H):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_cells_sorted: encoded cells, a list of batch_size tensors of dimension (num_cells, max_cell_length)
        :param tag_H: hidden state from TagDeoder, a list of batch_size tensors of dimension (num_cells, TagDecoder's decoder_dim)
        :param cell_lengths: caption lengths, a list of batch_size tensor of dimension (num_cells, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights
        """
        if hasattr(self, 'input_filter'):
            encoder_out = self.input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Flatten image
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Decode cell content
        predictions_cell = []
        alphas_cell = []
        decode_lengths_cell = []
        for i in range(batch_size):
            num_cells = cell_lengths[i].size(0)
            embeddings = self.embedding(encoded_cells_sorted[i])
            decode_lengths = (cell_lengths[i] - 1).tolist()
            max_decode_lengths = decode_lengths[0]
            predictions = torch.zeros(num_cells, max_decode_lengths, self.vocab_size).to(device)
            alphas = torch.zeros(num_cells, max_decode_lengths, num_pixels).to(device)
            if isinstance(self.decode_step_pred, nn.LSTMCell):
                h_LM, c_LM, h_cell, c_cell = self.init_hidden_state(encoder_out[i].unsqueeze(0), num_cells)
            else:
                h_LM, h_cell = self.init_hidden_state(encoder_out[i].unsqueeze(0), num_cells)
            for t in range(max_decode_lengths):
                batch_size_t = sum([l > t for l in decode_lengths])
                # Language LSTM
                if isinstance(self.decode_step_LM, nn.LSTMCell):
                    h_LM, c_LM = self.decode_step_LM(
                        embeddings[:batch_size_t, t, :],
                        (h_LM[:batch_size_t], c_LM[:batch_size_t]))  # (batch_size_t, decoder_dim)
                else:
                    h_LM = self.decode_step_LM(
                        embeddings[:batch_size_t, t, :],
                        h_LM[:batch_size_t])  # (batch_size_t, decoder_dim)

                # Attention
                attention_weighted_encoding, alpha = self.attention(
                    encoder_out[i].unsqueeze(0), tag_H[i][:batch_size_t],
                    h_LM)
                # Decoder LSTM
                gate = self.sigmoid(self.f_beta(h_cell[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                if isinstance(self.decode_step_pred, nn.LSTMCell):
                    h_cell, c_cell = self.decode_step_pred(
                        attention_weighted_encoding,
                        (h_cell[:batch_size_t], c_cell[:batch_size_t]))  # (batch_size_t, decoder_dim)
                else:
                    h_cell = self.decode_step_pred(
                        attention_weighted_encoding,
                        h_cell[:batch_size_t])  # (batch_size_t, decoder_dim)
                predictions[:batch_size_t, t, :] = self.fc(self.dropout(h_cell))  # (batch_size_t, vocab_size)
                alphas[:batch_size_t, t, :] = alpha
            predictions_cell.append(predictions)
            alphas_cell.append(alphas)
            decode_lengths_cell.append(decode_lengths)

        return predictions_cell, decode_lengths_cell, alphas_cell

class DualDecoder(nn.Module):
    """
    Dual decoder model:
        first decoder generates structure of the table
        second decoder generates cell content
    """
    def __init__(self, tag_attention_dim, cell_attention_dim, tag_embed_dim, cell_embed_dim,
                 tag_decoder_dim, language_dim, cell_decoder_dim,
                 tag_vocab_size, cell_vocab_size, td_encode,
                 decoder_cell=nn.LSTMCell, encoder_dim=512, dropout=0.5,
                 cell_decoder_type=1,
                 cnn_layer_stride=None, tag_H_grad=True, predict_content=True, predict_bbox=False):
        """
        :param tag_attention_dim: size of attention network for tags
        :param cell_attention_dim: size of attention network for cells
        :param tag_embed_dim: embedding size of tags
        :param cell_embed_dim: embedding size of cell content
        :param tag_decoder_dim: size of tag decoder's RNN
        :param language_dim: size of language model's RNN
        :param cell_decoder_dim: size of cell decoder's RNN
        :param tag_vocab_size: size of tag vocabulary
        :param cell_vocab_size: size of cellvocabulary
        :param td_encode: encodings for ('<td>', ' >')
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param mini_batch_size: batch size of cells to reduce GPU memory usage
        """
        super(DualDecoder, self).__init__()

        self.tag_attention_dim = tag_attention_dim
        self.cell_attention_dim = cell_attention_dim
        self.tag_embed_dim = tag_embed_dim
        self.cell_embed_dim = cell_embed_dim
        self.tag_decoder_dim = tag_decoder_dim
        self.language_dim = language_dim
        self.cell_decoder_dim = cell_decoder_dim
        self.tag_vocab_size = tag_vocab_size
        self.cell_vocab_size = cell_vocab_size
        self.decoder_cell = decoder_cell
        self.encoder_dim = encoder_dim
        self.dropout = dropout
        self.td_encode = td_encode
        self.tag_H_grad = tag_H_grad
        self.predict_content = predict_content
        self.predict_bbox = predict_bbox
        self.relu_tag = nn.ReLU()
        self.relu_cell = nn.ReLU()

        self.tag_decoder = TagDecoder(
            tag_attention_dim,
            tag_embed_dim,
            tag_decoder_dim,
            tag_vocab_size,
            td_encode,
            decoder_cell,
            encoder_dim,
            dropout,
            cnn_layer_stride['tag'] if isinstance(cnn_layer_stride, dict) else None,
            self.tag_H_grad)
        if cell_decoder_type == 1:
            self.cell_decoder = CellDecoder_baseline(
                cell_attention_dim,
                cell_embed_dim,
                tag_decoder_dim,
                cell_decoder_dim,
                cell_vocab_size,
                decoder_cell,
                encoder_dim,
                dropout,
                cnn_layer_stride['cell'] if isinstance(cnn_layer_stride, dict) else None)
        elif cell_decoder_type == 2:
            self.cell_decoder = CellDecoder(
                cell_attention_dim,
                cell_embed_dim,
                tag_decoder_dim,
                language_dim,
                cell_decoder_dim,
                cell_vocab_size,
                decoder_cell,
                encoder_dim,
                dropout,
                cnn_layer_stride['cell'] if isinstance(cnn_layer_stride, dict) else None)
        self.bbox_loss = BBoxLoss()
        self.cell_bbox_regressor = CellBBox(tag_decoder_dim)

        if torch.cuda.device_count() > 1:
            self.tag_decoder = MyDataParallel(self.tag_decoder)
            self.cell_decoder = MyDataParallel(self.cell_decoder)
            self.cell_bbox_regressor = MyDataParallel(self.cell_bbox_regressor)

    def load_pretrained_tag_decoder(self, tag_decoder):
        self.tag_decoder = tag_decoder

    def fine_tune_tag_decoder(self, fine_tune=False):
        for p in self.tag_decoder.parameters():
            p.requires_grad = fine_tune

    def inference(self, encoder_out, word_map,
                  max_steps={'tag': 400, 'cell': 200},
                  beam_size={'tag': 5, 'cell': 5},
                  return_attention=False):
        """
        Inference on test images with beam search
        """
        res = self.tag_decoder.inference(
            encoder_out,
            word_map['word_map_tag'],
            max_steps['tag'],
            beam_size['tag'],
            return_attention=return_attention
        )
        if res is not None:
            output, tag_H = res[:-1], res[-1]
            if self.predict_content:
                cell_res = self.cell_decoder.inference(
                    encoder_out,
                    tag_H,
                    word_map['word_map_cell'],
                    max_steps['cell'],
                    beam_size['cell'],
                    return_attention=return_attention
                )
                if return_attention:
                    cell_seqs, cell_alphas = cell_res
                    output += (cell_seqs, cell_alphas)
                else:
                    cell_seqs = cell_res
                    output += (cell_seqs,)
            if self.predict_bbox:
                cell_bbox = self.cell_bbox_regressor(
                    encoder_out,
                    tag_H
                )
                output += (cell_bbox,)
            return output
        else:
            return None

    def forward(self, encoder_out, encoded_tags_sorted, tag_lengths, cells=None, cell_lens=None, num_cells=None):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_tags_sorted: encoded tags, a tensor of dimension (batch_size, max_tag_length)
        :param tag_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :param encoded_cells: encoded cells, a list of batch_size tensors of dimension (num_cells, max_cell_length)
        :param cell_lengths: caption lengths, a list of batch_size tensor of dimension (num_cells, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights
        """
        batch_size = encoder_out.size(0)
        N_GPUS = torch.cuda.device_count()
        if N_GPUS > 1 and N_GPUS != batch_size:
            # WHen multiple GPUs are available, rearrange the samples
            # in the batch so that partition is more balanced. This
            # increases training speed and reduce the chance of
            # GPU memory overflow.
            balance_inds = np.arange(np.ceil(batch_size / N_GPUS) * N_GPUS, dtype=int).reshape(-1, N_GPUS).flatten('F')[:batch_size]
            encoder_out = encoder_out[balance_inds]
            encoded_tags_sorted = encoded_tags_sorted[balance_inds]
            tag_lengths = tag_lengths[balance_inds]
            if num_cells is not None:
                num_cells = num_cells[balance_inds]
            if self.predict_content:
                cells = [cells[ind] for ind in balance_inds]
                cell_lens = [cell_lens[ind] for ind in balance_inds]

        output = self.tag_decoder(
            encoder_out,
            encoded_tags_sorted,
            tag_lengths,
            num_cells=num_cells if self.predict_content or self.predict_bbox else None,
            max_tag_len=(tag_lengths[0] - 1).item()
        )

        if self.predict_content or self.predict_bbox:
            tag_H = output[-1]
            if self.predict_bbox:
                predictions_cell_bboxes = self.cell_bbox_regressor(
                    encoder_out,
                    tag_H
                )

            if self.predict_content:
                # Sort cells of each sample by decreasing length
                for j in range(len(cells)):
                    cell_lens[j], s_ind = cell_lens[j].sort(dim=0, descending=True)
                    cells[j] = cells[j][s_ind]
                    tag_H[j] = tag_H[j][s_ind]

                predictions_cell, decode_lengths_cell, alphas_cell = self.cell_decoder(
                    encoder_out,
                    cells,
                    cell_lens,
                    tag_H
                )

            output = output[:3]
            if self.predict_content:
                output += (predictions_cell, decode_lengths_cell, alphas_cell, cells)
            if self.predict_bbox:
                output += (predictions_cell_bboxes,)

        if N_GPUS > 1 and N_GPUS != batch_size:
            # Restore the correct order of samples in the batch to compute
            # the correct loss
            restore_inds = np.arange(np.ceil(batch_size / N_GPUS) * N_GPUS, dtype=int).reshape(N_GPUS, -1).flatten('F')[:batch_size]
            output = tuple([item[ind] for ind in restore_inds] if isinstance(item, list) else item[restore_inds] for item in output)
        return output

    def train_epoch(self, train_loader, encoder, criterion,
                    encoder_optimizer, tag_decoder_optimizer, cell_decoder_optimizer, cell_bbox_regressor_optimizer,
                    epoch, args):
        """
        Performs one epoch's training.

        :param train_loader: DataLoader for training data
        :param encoder: encoder model
        :param criterion: loss layer
        :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
        :param tag_decoder_optimizer: optimizer to update tag decoder's weights
        :param cell_decoder_optimizer: optimizer to update cell decoder's weights
        :param epoch: epoch number
        """
        self.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        losses_tag = AverageMeter()  # loss (per word decoded)
        losses_total = AverageMeter()  # loss (per word decoded)
        top1accs_tag = AverageMeter()  # top1 accuracy
        if self.predict_content:
            losses_cell = AverageMeter()  # loss (per word decoded)
            top1accs_cell = AverageMeter()  # top1 accuracy
        if self.predict_bbox:
            losses_cell_box = AverageMeter()  # top1 accuracy

        start = time.time()
        # Batches
        train_loader.shuffle()
        for i, batch in enumerate(train_loader):
            try:
                imgs, tags, tag_lens, num_cells = batch[:4]
                # Move to GPU, if available
                imgs = imgs.to(device)
                tags = tags.to(device)
                tag_lens = tag_lens.to(device)
                num_cells = num_cells.to(device)
                if self.predict_content:
                    cells, cell_lens = batch[4:6]
                    cells = [c.to(device) for c in cells]
                    cell_lens = [c.to(device) for c in cell_lens]
                else:
                    cells = None
                    cell_lens = None

                if self.predict_bbox:
                    cell_bboxes = batch[-1]
                    cell_bboxes = [c.to(device) for c in cell_bboxes]

                # Forward prop.
                imgs = encoder(imgs)

                # Flatten image
                batch_size = imgs.size(0)
                # encoder_dim = imgs.size(-1)
                # imgs = imgs.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
                tag_lens = tag_lens.squeeze(1)

                # Sort input data by decreasing tag lengths
                tag_lens, sort_ind = tag_lens.sort(dim=0, descending=True)
                imgs = imgs[sort_ind]
                tags_sorted = tags[sort_ind]
                num_cells = num_cells[sort_ind]
                if self.predict_content:
                    cells = [cells[ind] for ind in sort_ind]
                    cell_lens = [cell_lens[ind] for ind in sort_ind]
                if self.predict_bbox:
                    cell_bboxes = [cell_bboxes[ind] for ind in sort_ind]

                output = self(imgs, tags_sorted, tag_lens, cells, cell_lens, num_cells)

                scores_tag, decode_lengths_tag, alphas_tag = output[:3]
                if self.predict_content:
                    scores_cell, decode_lengths_cell, alphas_cell, cells = output[3:7]
                if self.predict_bbox:
                    predictions_cell_bboxes = output[-1]

                # Gather results to the same GPU
                if torch.cuda.device_count() > 1:
                    if self.predict_content:
                        for s, cell in zip(range(len(scores_cell)), cells):
                            if scores_cell[s].get_device() != cell.get_device():
                                scores_cell[s] = scores_cell[s].to(device)
                                alphas_cell[s] = alphas_cell[s].to(device)
                    if self.predict_bbox:
                        for s, cell_bbox in zip(range(len(predictions_cell_bboxes)), cell_bboxes):
                            if predictions_cell_bboxes[s].get_device() != cell_bbox.get_device():
                                predictions_cell_bboxes[s] = predictions_cell_bboxes[s].to(device)

                # Calculate tag loss
                targets_tag = tags_sorted[:, 1:]
                scores_tag = pack_padded_sequence(scores_tag, decode_lengths_tag, batch_first=True)[0]
                targets_tag = pack_padded_sequence(targets_tag, decode_lengths_tag, batch_first=True)[0]
                loss_tag = criterion['tag'](scores_tag, targets_tag)
                # Add doubly stochastic attention regularization
                # loss_tag += args.alpha_c * ((1. - alphas_tag.sum(dim=1)) ** 2).mean()
                loss_tag += args.alpha_tag * (self.relu_tag(1. - alphas_tag.sum(dim=1)) ** 2).mean()
                loss = args.tag_loss_weight * loss_tag
                top1_tag = accuracy(scores_tag, targets_tag, 1)
                tag_count = sum(decode_lengths_tag)
                losses_tag.update(loss_tag.item(), tag_count)
                top1accs_tag.update(top1_tag, tag_count)

                # Calculate cell loss
                if self.predict_content and args.cell_loss_weight > 0:
                    loss_cell = 0.
                    reg_alphas_cell = 0
                    for scores, gt, decode_lengths, alpha in zip(scores_cell, cells, decode_lengths_cell, alphas_cell):
                        targets = gt[:, 1:]
                        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
                        __loss_cell = criterion['cell'](scores, targets)
                        # __loss_cell += args.alpha_c * ((1. - alpha.sum(dim=1)) ** 2).mean()
                        reg_alphas_cell += args.alpha_cell * (self.relu_cell(1. - alpha.sum(dim=(0, 1))) ** 2).mean()
                        top1_cell = accuracy(scores, targets, 1)
                        cell_count = sum(decode_lengths)
                        losses_cell.update(__loss_cell.item(), cell_count)
                        top1accs_cell.update(top1_cell, cell_count)
                        loss_cell += __loss_cell
                    loss_cell /= batch_size
                    loss_cell += reg_alphas_cell / batch_size
                    loss += args.cell_loss_weight * loss_cell
                # Calculate cell bbox loss
                if self.predict_bbox and args.cell_bbox_loss_weight > 0:
                    loss_cell_bbox = 0.
                    for pred, gt in zip(predictions_cell_bboxes, cell_bboxes):
                        __loss_cell_bbox = self.bbox_loss(gt, pred)
                        losses_cell_bbox.update(__loss_cell_bbox.item(), pred.size(0))
                        loss_cell_bbox += __loss_cell_bbox
                    loss_cell_bbox /= batch_size
                    loss += args.cell_bbox_loss_weight * loss_cell_bbox

                losses_total.update(loss.item(), 1)

                # Back prop.
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()
                if tag_decoder_optimizer is not None:
                    tag_decoder_optimizer.zero_grad()
                if self.predict_content:
                    cell_decoder_optimizer.zero_grad()
                if self.predict_bbox:
                    cell_bbox_regressor_optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                if args.grad_clip is not None:
                    if encoder_optimizer is not None:
                        clip_gradient(encoder_optimizer, args.grad_clip)
                    if tag_decoder_optimizer is not None:
                        clip_gradient(tag_decoder_optimizer, args.grad_clip)
                    if self.predict_content:
                        clip_gradient(cell_decoder_optimizer, args.grad_clip)
                    if self.predict_bbox:
                        clip_gradient(cell_bbox_regressor_optimizer, args.grad_clip)

                # Update weights
                if encoder_optimizer is not None:
                    encoder_optimizer.step()
                if tag_decoder_optimizer is not None:
                    tag_decoder_optimizer.step()
                if self.predict_content:
                    cell_decoder_optimizer.step()
                if self.predict_bbox:
                    cell_bbox_regressor_optimizer.step()

                batch_time.update(time.time() - start)
                start = time.time()

                # Print status
                if i % args.print_freq == 0:
                    verbose = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)) + \
                              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time) + \
                              'Loss_total {loss_total.val:.4f} ({loss_total.avg:.4f})\t'.format(loss_total=losses_total) + \
                              'Loss_tag {loss_tag.val:.4f} ({loss_tag.avg:.4f})\t'.format(loss_tag=losses_tag) + \
                              'Acc_tag {top1_tag.val:.3f} ({top1_tag.avg:.3f})\t'.format(top1_tag=top1accs_tag)
                    if self.predict_content:
                        verbose += 'Loss_cell {loss_cell.val:.4f} ({loss_cell.avg:.4f})\t'.format(loss_cell=losses_cell) + \
                                   'Acc_cell {top1_cell.val:.3f} ({top1_cell.avg:.3f})\t'.format(top1_cell=top1accs_cell)
                    if self.predict_bbox:
                        verbose += 'Loss_cell_bbox {loss_cell_bbox.val:.4f} ({loss_cell_bbox.avg:.4f})\t'.format(loss_cell_bbox=losses_cell_bbox)

                    print(verbose, file=sys.stderr)
                    sys.stderr.flush()

                    batch_time.reset()
                    losses_total.reset()
                    losses_tag.reset()
                    top1accs_tag.reset()
                    if self.predict_content:
                        losses_cell.reset()
                        top1accs_cell.reset()
                    if self.predict_bbox:
                        losses_cell_bbox.reset()
            except Exception as e:
                raise
