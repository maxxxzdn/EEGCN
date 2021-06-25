# EEGCN: Classification of EEG signals into silence vs listening with Graph Neural Network (PyTorch)

**utils**: training (train) and evaluating (test) routines

**model**: contain classifiers to solve the problem (torch.nn.Module classes)

**main**: script to run, it loads dataset and performs training of the model on it.

* Run the training script:
``` 
python main.py --wandb 0 --exp_name 16 --epochs 100 --num_features 32 --batch_size 16 --hops 4 --autoencoder 'conv' --kernel_size 13
```

