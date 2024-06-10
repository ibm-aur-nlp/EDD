import os
import numpy as np
import torch
from torch import nn
from torch.nn.parallel._functions import Scatter, Gather
from PIL import Image, ImageOps
from math import ceil

def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            per_gpu = ceil(len(obj) / len(target_gpus))
            partition = [obj[k * per_gpu: min(len(obj), (k + 1) * per_gpu)] for k, _ in enumerate(target_gpus)]
            for i, target in zip(range(len(partition)), target_gpus):
                for j in range(len(partition[i])):
                    partition[i][j] = partition[i][j].to(torch.device('cuda:%d' % target))
            return partition
            # return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        if isinstance(out, list):
            return [item for output in outputs for item in output]
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None

class MyDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(MyDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(MyDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(out_dir, data_name, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer):
    """
    Saves model checkpoint.
    :param out_dir: output dir
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    """
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    try:
        if not os.path.exists(os.path.join(out_dir, data_name)):
            os.makedirs(os.path.join(out_dir, data_name))
        torch.save(state, os.path.join(out_dir, data_name, filename))
    except Exception:
        torch.save(state, os.path.join(os.environ['RESULT_DIR'], filename))

def save_checkpoint_dual(out_dir, data_name, epoch,
                         encoder, decoder, encoder_optimizer,
                         tag_decoder_optimizer, cell_decoder_optimizer,
                         cell_bbox_regressor_optimizer):
    """
    Saves EDD model checkpoint.
    :param out_dir: output dir
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param tag_decoder_optimizer: optimizer to update tag decoder's weights
    :param cell_decoder_optimizer: optimizer to update cell decoder's weights
    :param cell_bbox_regressor_optimizer: optimizer to update cell bbox regressor's weights
    """
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'tag_decoder_optimizer': tag_decoder_optimizer,
             'cell_decoder_optimizer': cell_decoder_optimizer,
             'cell_bbox_regressor_optimizer': cell_bbox_regressor_optimizer}
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    if not os.path.exists(os.path.join(out_dir, data_name)):
        os.makedirs(os.path.join(out_dir, data_name))
    torch.save(state, os.path.join(out_dir, data_name, filename))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def change_learning_rate(optimizer, new_lr):
    """
    Change learning rate.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param new_lr: new learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def image_resize(imagepath, image_size, keep_AR=True):
    with Image.open(imagepath) as im:
        old_size = im.size  # old_size[0] is in (width, height) format
        if keep_AR:
            ratio = float(image_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = im.resize(new_size, Image.Resampling.LANCZOS)
            delta_w = image_size - new_size[0]
            delta_h = image_size - new_size[1]
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            new_im = ImageOps.expand(im, padding)
        else:
            new_im = im.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return new_im, old_size

def image_rescale(imagepath, image_size, keep_AR=True, transpose=True, return_size=False):
    new_im, old_size = image_resize(imagepath, image_size, keep_AR)
    img = np.array(new_im)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if transpose:
        img = img.transpose(2, 0, 1)
    if return_size:
        return img, old_size
    else:
        return img

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


if __name__ == '__main__':
    pass
