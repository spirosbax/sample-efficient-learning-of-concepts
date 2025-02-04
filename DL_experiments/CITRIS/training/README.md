# Experiment run-files

This folder contains all files to start an experiment with the implemented methods.
Note that for any experiment, you first need to have generated a dataset.
See the files in the folder `data_generation/` for further information on the dataset generation.

## Causal Encoder

To train a causal encoder, you can use the following command:
```bash
python train_causal_encoder.py --data_dir ../data/causal3d/
```

## CITRIS-VAE

To train a CITRIS-VAE, you can use the following command:
```bash
python train_vae.py --model CITRISVAE \
                    --data_dir ../data/causal3d/ \
                    --causal_encoder_checkpoint ../data/causal3d/models/CausalEncoder.ckpt \
                    --num_latents 32 \
                    --beta_t1 1.0 \
                    --beta_classifier 1.0 \
                    --beta_mi_estimator 1.0 \
                    --graph_learning_method ENCO \
                    --lambda_sparse 0.02
```
For running NOTEARS, use `--graph_learning_method NOTEARS --lambda_sparse 0.002` instead.
