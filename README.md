# Beam Tracking Using Previously-Observed Beam Sequences
The baseline solution for the ViWi-Vision Aided Beam Tracking (ViWi-BT) task. Please see [ViWi Vision-Aided mmWave Beam Tracking: Dataset, Task, and Baseline Solutions](https://arxiv.org/abs/2002.02445) for more details.

# Dependecies
1) Python 3.7 and later.

2) PyTorch 1.3 and later (with torchvision).

3) Numpy 1.6 or later.

4) Pandas.

5) NVIDIA GPU with CUDA (10.XX).

# Running instructions
These scripts assume the ViWi-BT dataset in .csv formate. The visual data, traain_set.csv, and val_set.csv files all need to be included in the same root directory as the scripts. To run the script, do the following:

1) In the script main_beam.py, adjust the training and validation hyperparameters as needed.

2) Run main_beam.py

The results will be saved in a ".mat" file.

# Citation
If you plan to use these scripts or part of them, please cite the following paper:
```
@misc{alrabeiah2020viwi,
    title={ViWi Vision-Aided mmWave Beam Tracking: Dataset, Task, and Baseline Solutions},
    author={Muhammad Alrabeiah and Jayden Booth and Andrew Hredzak and Ahmed Alkhateeb},
    year={2020},
    eprint={2002.02445},
    archivePrefix={arXiv},
    primaryClass={eess.SP}
}
```
