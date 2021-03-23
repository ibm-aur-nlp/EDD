# Image-based table recognition: data, model, evaluation

## Task

Converting table images into HTML code

## Dataset

[PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) contains over 500k table
images annotated with the corresponding HTML representation.

## Model

Encoder-Dual-Decoder (EDD)

![Encoder-Dual-Decoder (EDD)](img/DualDecoderArch.png "Encoder-Dual-Decoder (EDD)")

## Evaluation

**T**ree-**E**dit-**D**istance-based **S**imilarity (TEDS)

`TEDS(T_1, T_2) = 1 - EditDistance(T_1, T_2) / max(|T_1|, |T_2|)`, where `EditDistance(T_1, T_2)` is the tree edit distance between `T_1` and `T_2`, and `|T|` is the number of nodes in `T`.

## Installation

Please use python 3 (>=3.6) environment.

`pip install -r requirements`

## Training and testing on PubTabNet

### Prepare data

Download PubTabNet and extract the files into the following file structure
```
{DATA_DIR}
|
-- train
   |
   -- PMCXXXXXXX.png
   -- ...
-- val
   |
   -- PMCXXXXXXX.png
   -- ...
-- test
   |
   -- PMCXXXXXXX.png
   -- ...
-- PubTabNet_2.0.0.jsonl
```

Prepare data for training
```
python prepare_data.py \
       --annotation {DATA_DIR}/PubTabNet_2.0.0.jsonl  \
       --image_dir {DATA_DIR} \
       --out_dir {TRAIN_DATA_DIR}
```

The following files will be generated in {TRAIN_DATA_DIR}:
```
- TRAIN_IMAGES_{POSTFIX}.h5          # Training images
- TRAIN_TAGS_{POSTFIX}.json          # Training structural tokens
- TRAIN_TAGLENS_{POSTFIX}.json       # Length of training structural tokens
- TRAIN_CELLS_{POSTFIX}.json         # Training cell tokens
- TRAIN_CELLLENS_{POSTFIX}.json      # Length of training cell tokens
- TRAIN_CELLBBOXES_{POSTFIX}.json    # Training cell bboxes
- VAL.json                           # Validation ground truth
- WORDMAP_{POSTFIX}.json             # Vocab
```
where `{POSTFIX}` is `PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size`
### Train tag decoder

Use larger (0.001) learning rate in the first 10 epochs
```
python train_dual_decoder.py \
       --out_dir {CHECKPOINT_DIR} \
       --data_folder {TRAIN_DATA_DIR} \
       --data_name PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size \
       --epochs 10 \
       --batch_size 10 \
       --fine_tune_encoder \
       --encoder_lr 0.001 \
       --fine_tune_tag_decoder \
       --tag_decoder_lr 0.001 \
       --tag_loss_weight 1.0 \
       --cell_decoder_lr 0.001 \
       --cell_loss_weight 0.0 \
       --tag_embed_dim 16 \
       --cell_embed_dim 80 \
       --encoded_image_size 28 \
       --decoder_cell LSTM \
       --tag_attention_dim 256 \
       --cell_attention_dim 256 \
       --tag_decoder_dim 256 \
       --cell_decoder_dim 512 \
       --cell_decoder_type 1 \
       --cnn_stride '{"tag":1, "cell":1}' \
       --resume
```

Use smaller (0.0001) learning rate for another 3 epochs
```
python train_dual_decoder.py \
       --out_dir {CHECKPOINT_DIR} \
       --data_folder {TRAIN_DATA_DIR} \
       --data_name PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size \
       --epochs 13 \
       --batch_size 10 \
       --fine_tune_encoder \
       --encoder_lr 0.0001 \
       --fine_tune_tag_decoder \
       --tag_decoder_lr 0.0001 \
       --tag_loss_weight 1.0 \
       --cell_decoder_lr 0.001 \
       --cell_loss_weight 0.0 \
       --tag_embed_dim 16 \
       --cell_embed_dim 80 \
       --encoded_image_size 28 \
       --decoder_cell LSTM \
       --tag_attention_dim 256 \
       --cell_attention_dim 256 \
       --tag_decoder_dim 256 \
       --cell_decoder_dim 512 \
       --cell_decoder_type 1 \
       --cnn_stride '{"tag":1, "cell":1}' \
       --resume
```

### Train dual decoders

**NOTE**:
- Sometimes when a random batch is too large, it may exceeds the GPU memory. When this happens, just re-execute the training command, which will resume from the latest checkpoint.
- Training dual decoders requires 2 V100 GPUs.

Use larger (0.001) learning rate in the first 10 epochs
```
python train_dual_decoder.py \
       --checkpoint {CHECKPOINT_DIR}/PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size/checkpoint_12.pth.tar \
       --out_dir {CHECKPOINT_DIR}/cell_decoder \
       --data_folder {TRAIN_DATA_DIR} \
       --data_name PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size \
       --epochs 23 \
       --batch_size 8 \
       --fine_tune_encoder \
       --encoder_lr 0.001 \
       --fine_tune_tag_decoder \
       --tag_decoder_lr 0.001 \
       --tag_loss_weight 0.5 \
       --cell_decoder_lr 0.001 \
       --cell_loss_weight 0.5 \
       --tag_embed_dim 16 \
       --cell_embed_dim 80 \
       --encoded_image_size 28 \
       --decoder_cell LSTM \
       --tag_attention_dim 256 \
       --cell_attention_dim 256 \
       --tag_decoder_dim 256 \
       --cell_decoder_dim 512 \
       --cell_decoder_type 1 \
       --cnn_stride '{"tag":1, "cell":1}' \
       --resume \
       --predict_content
```

Use smaller (0.0001) learning rate for another 2 epochs
```
python train_dual_decoder.py \
       --out_dir {CHECKPOINT_DIR}/cell_decoder \
       --data_folder {TRAIN_DATA_DIR} \
       --data_name PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size \
       --epochs 25 \
       --batch_size 8 \
       --fine_tune_encoder \
       --encoder_lr 0.0001 \
       --fine_tune_tag_decoder \
       --tag_decoder_lr 0.0001 \
       --tag_loss_weight 0.5 \
       --cell_decoder_lr 0.0001 \
       --cell_loss_weight 0.5 \
       --tag_embed_dim 16 \
       --cell_embed_dim 80 \
       --encoded_image_size 28 \
       --decoder_cell LSTM \
       --tag_attention_dim 256 \
       --cell_attention_dim 256 \
       --tag_decoder_dim 256 \
       --cell_decoder_dim 512 \
       --cell_decoder_type 1 \
       --cnn_stride '{"tag":1, "cell":1}' \
       --resume \
       --predict_content
```


### Inferencing

Get validation performance
```
python eval.py \
       --image_folder {DATA_DIR}/val \
       --result_json {RESULT_DIR}/RESULT_FILE.json \
       --gt {TRAIN_DATA_DIR}/VAL.json \
       --model {CHECKPOINT_DIR}/cell_decoder/PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size/checkpoint_24.pth.tar \
       --word_map {TRAIN_DATA_DIR}/WORDMAP_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json \
       --image_size 448 \
       --dual_decoder \
       --beam_size '{"tag":3, "cell":3}' \
       --max_steps '{"tag":1800, "cell":600}'
```
This will save the TEDS score of every validation sample in `{RESULT_DIR}/RESULT_FILE.json` in the following format:
```
{
  'PMCXXXXXXX.png': float,
}
```

Get testing performance
```
python eval.py \
       --image_folder {DATA_DIR}/test \
       --result_json {RESULT_DIR}/RESULT_FILE.json \
       --model {CHECKPOINT_DIR}/cell_decoder/PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size/checkpoint_24.pth.tar \
       --word_map {TRAIN_DATA_DIR}/WORDMAP_PubTabNet_False_keep_AR_300_max_tag_len_100_max_cell_len_512_max_image_size.json \
       --image_size 448 \
       --dual_decoder \
       --beam_size '{"tag":3, "cell":3}' \
       --max_steps '{"tag":1800, "cell":600}'
```
This will save the inference result (HTML code) of every testing sample in `{RESULT_DIR}/RESULT_FILE.json` in the following format:
```
{
  'PMCXXXXXXX.png': str,
}
```
The json file can be compared agains the ground truth using the code [here](https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src). The ground truth of test set has been kept secret.

## Cite us

```
@article{zhong2019image,
  title={Image-based table recognition: data, model, and evaluation},
  author={Zhong, Xu and ShafieiBavani, Elaheh and Yepes, Antonio Jimeno},
  journal={arXiv preprint arXiv:1911.10683},
  year={2019}
}
```
