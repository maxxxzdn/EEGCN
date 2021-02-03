# EEGCN: Classification of EEG signals into silence vs listening with Graph Neural Network (PyTorch)

**utils**: training (train) and evaluating (test) routines

**model**: contains the model

**main**: script to run, it creates dataset and performs training of the model on it.

* Run the training script:
``` 
python3 main.py --graph_info ../EEG_data/edge_index3.txt --hops 1
```

