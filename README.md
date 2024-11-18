# BioRGM

## Issues
- [ ] Introduce model evaluation
- [x] Load Model from checkpoints or initialize random model from jupyter notebook for evaluation
- [x] Integrate wandb
- [x] Implement checkpoints
- [x] Track triplets generated
- [x] Combine EmbeddingModel and GINModel

## Backlog
- [ ] Track augmentations performed for each augmentation
- [ ] Parallelize creation of InMemory PubchemDataset
- [ ] Add another 50 augmentations
- [ ] Configure logging
- [ ] Set up pytest
- [ ] InMemoryDataset should not create "pre-filter.pt" and "pre_transform.pt" files.