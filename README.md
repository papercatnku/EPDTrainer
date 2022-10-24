# EPDTrainer(Effortless Paradigm Driven Trainer) is a training engine designed for make Training CNN/DL Models easily.

## Content
- Design Notes
- Setup
- Usage
- Demo

## Design pricinpls
- Training procedures are much the same(and tedious) for volatile tasks. To make training models easier and easy to "plug and play" several training paradigms should dedicately designed. And those paradigms are cores of training tools.

- When code need to be channged, try the best to avoid write code with same functions. It's not neccessary but a best practice for experienced engineers.

- Visualizing training details is neccessary for experiments, especially for noveltechs. Tensorboard/wandb/... may be helpful tools.

- For embedding engineers, following features are which unfortunatly neglected by several ongoing similar systems
  - Port Model friendly   
    => my solution is to seperate model into inference and decoding part. inference model are designed to be easily exported. Decoding part is flexible
  - Quantization aware training.
  - Avoid overencapsulation for balance between productivity and modifiablility.

## Setup   
    To be filled


