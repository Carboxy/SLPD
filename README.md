SLPD: Slide-level Prototypical Distillation for WSIs
===========
<details>
<summary>
  <b>SLPD: Slide-level Prototypical Distillation for WSIs</b>, MICCAI 2023.
  <a href="https://arxiv.org/abs/2307.10696" target="blank">[arXiv]</a>
</summary>

</details>

<div align="center">
  <img width="100%" alt="SLPD Illustration" src=".SLPD_framework.PNG">
</div>

<details>
  <summary>
	  <b>Highlights</b>
  </summary>

1. **Slide-Level Representation Learning:** Aiming towards slide-level representations, we propose Slide-Level Prototypical Distillation (SLPD) to explore intra- and inter-slide semantic structures for context modeling on WSIs. Specifically, we iteratively perform intra-slide clustering for the regions (4096×4096 patches) within each WSI to yield the prototypes and encourage the region representations to be closer to the assigned prototypes. By representing each slide with its prototypes, we further select similar slides by the set distance of prototypes and assign the regions by cross-slide prototypes for distillation.
2. **Exploring Semantic Structures of Slides:**  We iteratively perform intra-slideclustering for the regions (4096×4096 patches) within each WSI to yield the prototypes and encourage the region representations to be closer to the assigned prototypes. By representing each slide with its prototypes, we further select similar slides by the set distance of prototypes and assign the regions by cross-slide prototypes for distillation. 
3. **SOTA performance:** The proposed method achieves state-of-the-art performance on multiple pathological datasets and tasks.
</details>

## Updates / TODOs
Please follow this GitHub for more updates.
- [ ] Release the pre-processed region-level features.


## Installation
- This repository is based on [HIPT](https://github.com/mahmoodlab/HIPT). Please refer to it for installation and data preparation. We will also release the pre-processed region-level features.
- We also use Redis to accelerate training 
  ```
  # install redis
  sudo apt install redis-server
  pip install redis
  
  # run redis
  tmux new -s redis
  redis-server
  ```

## Usage 
### Load Data into Redis
Run the Notebook `Hierarchical-Pretraining/extract_features/feature_to_redis_brca.ipynb` or `Hierarchical-Pretraining/extract_features/feature_to_redis_lung.ipynb` to load data into Redis, which can obviously accelerate training.


### SLPD Pre-Training
- For pre-training on NSCLS dataset
  ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 Hierarchical-Pretraining/main_SLPD.py --output_dir OUTPUT --dataset_dir Hierarchical-Pretraining/extract_features/tcga_lung_4k.json --num_cluster 4 --min_num_region 32
  ```
- For pre-training on BRCA dataset
  ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=8888 --nproc_per_node=4 Hierarchical-Pretraining/main_SLPD.py --output_dir OUTPUT --dataset_dir Hierarchical-Pretraining/extract_features/tcga_brca_4k.json --num_cluster 2 --min_num_region 16
  ```

### Weakly-Supervised Training & Evaluation
We follow [HIPT](https://github.com/mahmoodlab/HIPT) to perform weakly-Supervised training and evaluation.

## Acknowledgement
We would like to thank the [HIPT](https://github.com/mahmoodlab/HIPT) for its insightful work and code base.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is TBD.

© This code is made available under the Commons Clasuse License and is available for non-commercial academic purposes.

