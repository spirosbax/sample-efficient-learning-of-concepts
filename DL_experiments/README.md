# VAE experiments

The VAE experiments are split up into 2 repositories, `DMS_VAE_experiments` contains the experiments
of the Action/Temporal sparsity datasets and `CITRIS` contains the experiments
for the Temporal Causal3DIdent dataset. Most of the code in those repositories is 
taken from their original repositories, which are [Temporal/Action](https://github.com/slachapelle/disentanglement_via_mechanism_sparsity/tree/main)
and [CITIRIS](https://github.com/phlippe/CITRIS/tree/main).
In each of directories we added a `test.py` script that runs the experiments. 
Before each experiment can be run, the models have to be trained

## Action/Temporal sparsity dataset Model Training

### Installing all dependencies
For these set of experiments, Python 3.7 has to be chosen and we provide a `requirements.txt` that is adapted
from the original repository. 

### Running the experiments

The `train_models.job` file contains all the commands used to train the models. The training can take up to 12 hours. 

After all models are run, the experiments can be run by 

```
python -u test.py --cluster --estimator --baseline
```

Running all experiments can take up to 20 hours. 

## Temporal Causal3DIdent Model Training

The data that was used can be dowloaded from [[LINK]](https://zenodo.org/records/6637749#.YqcWCnVBxCA)
and have to be put into the `data` folder. 

### Installing all dependencies

We provide an environment file, that is adapted from the original repository. To insall the environment
run
```bash
conda env create -f environment.yml
```

### Running the experiments

The training in this case is a bit more involved. The following scripts or the commands contained in them, have 
to be run in order. 
```bash
train_auto_encoder.job
train_citrisvae.job
train_ivae.job
```
Training all models can take up to 24 hours. 

After all models are run, the experiments can be run by 
```
python -u test.py --cluster --estimator --baseline
```
Running all experiments can take up to 20 hours. 
