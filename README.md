# EEGCN: Classification of EEG signals into silence vs listening with Graph Neural Network (PyTorch)

**utils**: training (train) and evaluating (test) routines

**model**: contain classifiers to solve the problem (torch.nn.Module classes)

**main**: script to run, it loads dataset and performs training of the model on it.

**baselines**: baseline models (EEGNet and ShallowNet)

* Run the training script:
``` 
python main.py --wandb 1 --key 0 --model GCN --epochs 300 --n_cnn 3 --n_mp 1 --d_latent 32 --d_hidden 64 --kernel_size 31 --p_dropout 0.4 --norm_proc batch
```

