---
base_model: HuggingFaceTB/SmolLM-135M
datasets:
- generator
library_name: peft
license: apache-2.0
tags:
- trl
- sft
- generated_from_trainer
model-index:
- name: SmolLM-135M-Instruct-Finetune-LoRA
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# SmolLM-135M-Instruct-Finetune-LoRA

This model is a fine-tuned version of [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) on the generator dataset.
It achieves the following results on the evaluation set:
- Loss: 1.4752

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 2503
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 4
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.9806        | 0.6173 | 25   | 1.9171          |
| 1.8272        | 1.2346 | 50   | 1.7049          |
| 1.6412        | 1.8519 | 75   | 1.5713          |
| 1.5453        | 2.4691 | 100  | 1.5098          |
| 1.5006        | 3.0864 | 125  | 1.4822          |
| 1.4829        | 3.7037 | 150  | 1.4752          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.43.3
- Pytorch 2.4.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1