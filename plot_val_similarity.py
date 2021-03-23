import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description='Plot validation performance of epochs')
parser.add_argument('--structure_only', action='store_true')
parser.add_argument('--model', type=str)
args = parser.parse_args()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cell_start_epoch = 12
steps = []
grid = []
span = []
all = []

root_dir = 'structure_only' if args.structure_only else 'full_table'
path = os.path.join('data/pubmed_dual', root_dir, args.model)
for epoch in range(40):
    try:
        with open(os.path.join(path, 'mini_val_%s.pkl' % (epoch)), 'rb') as remote_file:
            results = pickle.load(remote_file)
            similarity = np.array(results['similarity'])
            n_grid_samples = results['n_grid_samples']
            n_span_samples = results['n_span_samples']
        steps.append(epoch + 1)
        grid.append(np.mean(similarity[:n_grid_samples]))
        span.append(np.mean(similarity[n_grid_samples:]))
        all.append(np.mean(similarity))
    except Exception:
        continue

ax[0].plot(steps, grid, 'b.-', label='Simple grid samples')
ax[0].plot(steps, span, 'r.-', label='Multi-row/col samples')
ax[0].plot(steps, all, 'k.-', label='All samples')
ax[0].set_ylim([round(min(span), 2) - 0.01, 1.0])
# ax[0].set_ylim([0.75, 1.0])
# ax.set_xticks(steps)
# if max(steps) >= cell_start_epoch:
#     ax.axvline(cell_start_epoch, 0, 1, color='k', linestyle='--')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Similarity')
ax[0].legend()
ax[0].grid()

over90 = np.sum(similarity[:n_grid_samples] >= 0.9)
hist, bin_edges = np.histogram(similarity[:n_grid_samples], bins=50, range=(0, 1))
ax[1].fill_between(bin_edges[1:], hist, 0, color='b',
                   alpha=0.5, linestyle='-', label='Simple grid samples (%.1f%% ≥ 0.9)' % (100. * over90 / n_grid_samples))
over90 = np.sum(similarity[n_grid_samples:] >= 0.9)
hist, bin_edges = np.histogram(similarity[n_grid_samples:], bins=50, range=(0, 1))
ax[1].fill_between(bin_edges[1:], hist, 0, color='r',
                   alpha=0.5, linestyle='-', label='Multi-row/col samples (%.1f%% ≥ 0.9)' % (100. * over90 / n_span_samples))
ax[1].legend(loc=2)
ax[1].set_xlabel('Similarity')
ax[1].set_ylabel('Occurance')
ax[1].set_xlim([0, 1])

plt.tight_layout()
plt.show()
