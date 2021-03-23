import torch
import h5py
import json
import os
import numpy as np
import random

class TableDatasetEvenLength(object):
    """
    Data loader for training baseline encoder-decoder model (WYGIWYS, Dent et al. 2017)
    """

    def __init__(self, data_folder, data_name, batch_size, transform=None):
        # Open hdf5 file where images are stored
        f = os.path.join(data_folder, 'TRAIN_IMAGES_' + data_name + '.hdf5')
        self.h = h5py.File(f, 'r')

        self.imgs = self.h['images']

        # Load encoded tables (completely into memory)
        with open(os.path.join(data_folder, 'TRAIN_TABLES_' + data_name + '.json'), 'r') as j:
            self.tables = json.load(j)

        # Load table lengths (completely into memory)
        with open(os.path.join(data_folder, 'TRAIN_TABLELENS_' + data_name + '.json'), 'r') as j:
            self.tablelens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.batch_size = batch_size
        self.batch_id = 0

    def shuffle(self):
        self.batch_id = 0
        self.batches = [[]]
        len_dict = dict()
        # Split samples into groups by table lengths
        for i, l in enumerate(self.tablelens):
            if l in len_dict:
                len_dict[l].append(i)
            else:
                len_dict[l] = [i]
        # Fill with long samples first, so that the samples do not need to be sorted before training
        lens = sorted(list(len_dict.keys()), key=lambda x: -x)
        # Shuffle each group
        for l in lens:
            random.shuffle(len_dict[l])
        # Generate batches
        for l in lens:
            k = 0
            # Fill previous incomplete batch
            if len(self.batches[-1]) < self.batch_size:
                deficit = min(len(len_dict[l]), self.batch_size - len(self.batches[-1]))
                self.batches[-1] += len_dict[l][k:k + deficit]
                k = deficit
            # Generate complete batches
            while len(len_dict[l]) - k >= self.batch_size:
                self.batches.append(len_dict[l][k:k + self.batch_size])
                k += self.batch_size
            # Create an incomplete batch with left overs
            if k < len(len_dict[l]):
                self.batches.append(len_dict[l][k:])
        # Shuffle the order of batches
        random.shuffle(self.batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_id < len(self.batches):
            samples = self.batches[self.batch_id]
            image_size = self.imgs[samples[0]].shape
            imgs = torch.zeros(len(samples), image_size[0], image_size[1], image_size[2], dtype=torch.float)
            table_size = len(self.tables[samples[0]])
            tables = torch.zeros(len(samples), table_size, dtype=torch.long)
            tablelens = torch.zeros(len(samples), 1, dtype=torch.long)
            for i, sample in enumerate(samples):
                img = torch.FloatTensor(self.imgs[sample] / 255.)
                if self.transform is not None:
                    imgs[i] = self.transform(img)
                else:
                    imgs[i] = img
                tables[i] = torch.LongTensor(self.tables[sample])
                tablelens[i] = torch.LongTensor([self.tablelens[sample]])
            self.batch_id += 1
            return imgs, tables, tablelens
        else:
            raise StopIteration()

    def __len__(self):
        return len(self.batches)

class TagCellDataset(object):
    """
    Data loader for training encoder-dual-decoder model
    """

    def __init__(self, data_folder, data_name, split, batch_size, mode='all', transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param batch_size: batch size
        :param mode: 'tag', 'tag+cell', 'tag+bbox', or 'tag+cell+bbox'
        :param transform: image transform pipeline
        """

        assert split in {'TRAIN', 'VAL', 'TEST'}
        assert mode in {'tag', 'tag+cell', 'tag+bbox', 'tag+cell+bbox'}

        self.split = split
        self.mode = mode
        self.batch_size = batch_size

        # Open hdf5 file where images are stored
        f = os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5')
        self.h = h5py.File(f, 'r')
        self.imgs = self.h['images']

        # Load encoded tags (completely into memory)
        with open(os.path.join(data_folder, self.split + '_TAGS_' + data_name + '.json'), 'r') as j:
            self.tags = json.load(j)

        # Load tag lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_TAGLENS_' + data_name + '.json'), 'r') as j:
            self.taglens = json.load(j)

        # Load cell lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CELLLENS_' + data_name + '.json'), 'r') as j:
            self.celllens = json.load(j)

        if 'cell' in self.mode:
            # Load encoded cell tokens (completely into memory)
            with open(os.path.join(data_folder, self.split + '_CELLS_' + data_name + '.json'), 'r') as j:
                self.cells = json.load(j)

        if 'bbox' in self.mode:
            # Load encoded tags (completely into memory)
            with open(os.path.join(data_folder, self.split + '_CELLBBOXES_' + data_name + '.json'), 'r') as j:
                self.cellbboxes = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.tags)
        self.ind = np.array(range(self.dataset_size))
        self.pointer = 0

    def shuffle(self):
        self.ind = np.random.permutation(self.dataset_size)
        self.pointer = 0

    def __iter__(self):
        return self

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i])
        tags = self.tags[i]
        taglens = self.taglens[i]
        cells = self.cells[i]
        celllens = self.celllens[i]
        image_size = self.imgsizes[i]
        return img, tags, taglens, cells, celllens, image_size

    def __next__(self):
        if self.pointer < self.dataset_size:
            if self.dataset_size - self.pointer >= self.batch_size:
                step = self.batch_size
                samples = self.ind[self.pointer:self.pointer + step]
            else:
                step = self.dataset_size - self.pointer
                lack = self.batch_size - step
                samples = np.hstack((self.ind[self.pointer:self.pointer + step], np.array(range(lack))))
            image_size = self.imgs[samples[0]].shape
            imgs = torch.zeros(len(samples), image_size[0], image_size[1], image_size[2], dtype=torch.float)
            max_tag_len = max([self.taglens[sample] for sample in samples])
            tags = torch.zeros(len(samples), max_tag_len, dtype=torch.long)
            taglens = torch.zeros(len(samples), 1, dtype=torch.long)
            num_cells = torch.zeros(len(samples), 1, dtype=torch.long)
            if 'cell' in self.mode:
                cells = []
                celllens = []
            if 'bbox' in self.mode:
                cellbboxes = []

            for i, sample in enumerate(samples):
                img = torch.FloatTensor(self.imgs[sample] / 255.)
                if self.transform is not None:
                    imgs[i] = self.transform(img)
                else:
                    imgs[i] = img
                tags[i] = torch.LongTensor(self.tags[sample][:max_tag_len])
                taglens[i] = torch.LongTensor([self.taglens[sample]])
                num_cells[i] = len(self.celllens[sample])
                if 'cell' in self.mode:
                    max_cell_len = max(self.celllens[sample])
                    cells.append(torch.LongTensor(self.cells[sample])[:, :max_cell_len])
                    celllens.append(torch.LongTensor(self.celllens[sample]))
                if 'bbox' in self.mode:
                    cellbboxes.append(torch.FloatTensor(self.cellbboxes[sample]))

            self.pointer += step
            output = (imgs, tags, taglens, num_cells)
            if 'cell' in self.mode:
                output += (cells, celllens)
            if 'bbox' in self.mode:
                output += (cellbboxes,)
            return output
        else:
            raise StopIteration()

    def __len__(self):
        return int(np.ceil(self.dataset_size / self.batch_size))
