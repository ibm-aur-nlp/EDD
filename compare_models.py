import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description='Plot validation performance of epochs')
parser.add_argument('--structure_only', action='store_true')
args = parser.parse_args()

if args.structure_only:
    models = ['DualDecoder_S1/3', 'DualDecoder_S2/3', 'DualDecoder_S1S1/3', 'DualDecoder_S2S1/3', 'DualDecoder_S2S2/3']
else:
    models = ['DualDecoder_S1S1', 'DualDecoder_S2S1']

fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
steps = {model: [] for model in models}
grid = {model: [] for model in models}
span = {model: [] for model in models}
whole = {model: [] for model in models}

for model in models:
    for epoch in range(40):
        try:
            with open(os.path.join('eval', model, 'Epoch_%d' % epoch, '3', 'val', 'results.pkl'), 'rb') as remote_file:
                results = pickle.load(remote_file)
                similarity = results['similarity']
                n_grid_samples = results['n_grid_samples']
                n_span_samples = results['n_span_samples']
            steps[model].append(epoch + 1)
            grid[model].append(np.mean(similarity[:n_grid_samples]))
            span[model].append(np.mean(similarity[n_grid_samples:]))
            whole[model].append(np.mean(similarity))
        except Exception:
            pass
    steps[model] = np.array(steps[model])

for model in models:
    ax[0].plot(steps[model] - steps[model][0] + 1, grid[model], '.-', label=model.split('/')[0])
    ax[0].set_title('Simple grid samples')
    ax[1].plot(steps[model] - steps[model][0] + 1, span[model], '.-', label=model.split('/')[0])
    ax[1].set_title('Multi-row/col samples')
    ax[2].plot(steps[model] - steps[model][0] + 1, whole[model], '.-', label=model.split('/')[0])
    ax[2].set_title('All samples')


for a in ax:
    a.set_ylim([round(min([min(span[model]) for model in models]), 2) - 0.01, 1.0])
    a.set_xlabel('Epochs')
    a.set_ylabel('Similarity')
    a.legend(loc=4)
    a.grid()
plt.tight_layout()
plt.show()
