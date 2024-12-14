# Learning Fair Representation: Demographic Parity vs Exempt and Non-Exempt Disparity Notations

This project has been build on top of the implementation code of the Learning Adversarially Fair and Transferable Representations paper ([project link](https://github.com/VectorInstitute/laftr/tree/master)). In this project, we implemented and modified the original code to include fairness metric I(hat Y;A|Xc) for representation learning, where Xc is a pre-defined binary critical feature.

We have included the following modifications:

1. We modified the main files so that one can directly specify which type of experiment (Demographic Parity or Non-exempt Disparity experiment) from the terminal input.
2. We modified the trainer object to provide a plot of the training losses at the end of the training in the experiment file.
3. We modified the dataset loader to take into account the existence of the feature xc for the Non-exempt Disparity experiment.
4. We created a new trainer function for the trainer object to take into account the existence of the feature xc for training.
5. We modified the tester to evaluate the Non-exempt Disparity measure.
6. We created a new model class inspired by the model class "WeightedEqoddsWassGan" and called it "WeightedEqoddsWassGanNEW," which considers the feature xc as an input to the adversary. (This also included the creation of new parent model classes for the model WeightedEqoddsWassGanNEW.)
7. Added new configuration files of type .json that contain specifications about the dataset ACSIncome and the training specifications.
For more details, please see ([Project repository](https://github.com/VectorInstitute/laftr/tree/master)).

Note: Comments that explain our modifications are written as in the example below:
```
####################### DatasetNEW is a new dataset handler that takes Xc into account #######################
from codebase.datasets import Dataset, DatasetNEW 
```

We recommend the user to check this notebook: (https://colab.research.google.com/drive/1biaqO-M8rKWTOvU8d_lFaD7RPrks_JF7?usp=sharing). 
This notebook includes step-by-step instructions to run the project on CoLab including dataset preprocessing and the terminal commands generator. 


Exact details about modifications are as follows. Modified files are:

* config.py: class ConfigParseNEW
* run_laftr.py: We deleted the part related to checking if the model has already been trained or not. Training is always activated.
* run_unf_clf.py: We deleted the part related to checking if the model has already been trained or not. Training is always activated.
* laftr.py: We added checks to the type of the experiment and specified the dataset loader, weights, training function, and evaluation  function based on the type of the experiment. 
* Transfer_learn.py: We added checks to the type of the experiment and specified the dataset loader, weights, training function, and evaluation  function based on the type of the experiment.  We load Xc if the experiment with non-exempt disparity. 
* datasets.py:  we added class DatasetNEW and class TransferDatasetNEW
* models.py: we added class AbstractBaseNetNEW, class DemParGanNEW, class EqOddsUnweightedGanNEW, class WassGanNEW, class WeightedGanNEW, class WeightedDemParGanNEW, class WeightedEqoddsGanNEW, class WeightedEqoddsWassGanNEW, class RegularizedFairClassifierNEW (Note we * were able to avoid adding all these classes. However, we did not want to make any modifications that might cause a conflict with the original code.
* trainer.py: we added new functions trainNew, and plot_metrics, and modified the train function
* tester.py: We added a new function evaluateNEW



## Installation
To run the project, clone it using:
```
git clone https://github.com/SokratALDARMINI/Representation_Learning.git
cd Representation_Learning
```
The project supports Python 3.6.X. So, configure the environment using Python 3.6. Then install the required dependencies:
```
pip install MarkupSafe==1.1.1
pip install absl-py==0.2.2
pip install astor==0.7.1
pip install gast==0.2.0
pip install grpcio==1.13.0
pip install Jinja2==2.10
pip install Markdown==2.6.11
pip install numpy==1.14.5
pip install protobuf==3.6.0
pip install six==1.11.0
pip install tensorboard==1.9.0
pip install tensorflow-gpu==1.9.0
pip install tensorflow==1.9.0
pip install termcolor==1.1.0
pip install Werkzeug==0.14.1
pip install matplotlib==3.3.0
pip install ipykernel
```
Before running the project, please preprocess the datasets and include them in the required directories (Preprocessing of the dataset has been conducted using the notebook mentioned above:  (https://colab.research.google.com/drive/1biaqO-M8rKWTOvU8d_lFaD7RPrks_JF7?usp=sharing)
1. /Representation_Learning/data/adult/ for Adult dataset with the name adult.npz
2. /Representation_Learning/data/ACSIncome/ for ACSIncome dataset with the name ACSIncome.npz

Once the dependencies are installed, one can run the project using commands of the following form:
```
python src/run_laftr.py conf/transfer/laftr_then_naive.json -o exp_name="laftr_example/Adult_Exp_1",train.n_epochs=100,train.aud_steps=2,train.batch_size=128,model.recon_coeff=1,model.fair_coeff=0.1,optim.learning_rate=0.0005,transfer.n_epochs=100 -n new=False,index=35 --data adult --dirs local
python src/run_unf_clf.py conf/transfer/laftr_then_naive.json -o exp_name="laftr_example/Adult_Exp_1/Exp_1_classification_transfer",train.n_epochs=100,train.aud_steps=2,train.batch_size=128,model.recon_coeff=1,model.fair_coeff=0.1,optim.learning_rate=0.0005,transfer.n_epochs=100,transfer.epoch_number=50 -n new=False,index=35 --data adult --dirs local
```

The first command trains the encoder, decoder, and classifier, to learn the fair representation. This command also generates a new representation for the test dataset in the checkpoints and saves it in the experiments folder. The second train a naive classifier on one of the checkpoints generated representation of the test dataset. The parameters for the two commands are as follows

1. Exp_name: Name of the experiment, used for logging and saving purposes

2. data: Dataset being used for the experiment ('adult' or 'ACSIncome')

3. num_experiment: the number of experiments to run. It will be used to distinguish between experiments with different values of the fair_coeff (\gamma).

4. train_epochs: Number of epochs to train the model (for representation learning part)

5. transfer_epochs: Number of epochs for training the naive classifier on the representation.

6. aud_steps: Number of adversarial update steps (the adversary parameters can be updated for more than one step while the encoder, decoder, and the classifier parameters are updated for one step)

7. batch_size: Size of each batch for training


8. recon_coeff: Coefficient for the reconstruction loss term


9. fair_coeff: Coefficient for the fairness loss term (\gamma)


10. learning_rate: Learning rate for the optimizer

11. transfer_epoch_number: Epoch number after which transfer learning starts
transfer_epoch_number = 950

12. New: Flag indicating whether the experiment is NEw (what we implemented with \Delta_{NE}) or old with from the original code.


13. index: Index of the feature to be used for non-exempt discrimination measure. (it is 35 for the Adult dataset and 10 for the ACSIncome dataset.)

To facilitate the generation of the commands, one can use the following script: 
```
def generate_experiment_commands(Exp_name,num_experiments, train_epochs, transfer_epochs, aud_steps, batch_size, recon_coeff, fair_coeff, learning_rate, transfer_epoch_number, data, New, index):
    commands = []
    i = num_experiments
    exp_name =Exp_name+ f"_Exp_{i}"
    if New:
      command1 = (
            f"python src/run_laftr.py conf/transfer/laftr_then_naive.json "
            f"-o exp_name=\"laftr_example/{exp_name}\","
            f"train.n_epochs={train_epochs},"
            f"train.aud_steps={aud_steps},"
            f"train.batch_size={batch_size},"
            f"model.class=WeightedEqoddsWassGanNEW,"
            f"model.recon_coeff={recon_coeff},"
            f"model.fair_coeff={fair_coeff},"
            f"optim.learning_rate={learning_rate},"
            f"transfer.n_epochs={transfer_epochs} "
            f"-n new={New},"
            f"index={index} "
            f"--data {data} --dirs local"
        )

      command2 = (
            f"python src/run_unf_clf.py conf/transfer/laftr_then_naive.json "
            f"-o exp_name=\"laftr_example/{exp_name}/Exp_{i}_classification_transfer\","
            f"train.n_epochs={train_epochs},"
            f"train.aud_steps={aud_steps},"
            f"train.batch_size={batch_size},"
            f"model.class=WeightedEqoddsWassGanNEW,"
            f"model.recon_coeff={recon_coeff},"
            f"model.fair_coeff={fair_coeff},"
            f"optim.learning_rate={learning_rate},"
            f"transfer.n_epochs={transfer_epochs},"
            f"transfer.epoch_number={transfer_epoch_number} "
            f"-n new={New},"
            f"index={index} "
            f"--data {data} --dirs local"
        )

    else:
      command1 = (
            f"python src/run_laftr.py conf/transfer/laftr_then_naive.json "
            f"-o exp_name=\"laftr_example/{exp_name}\","
            f"train.n_epochs={train_epochs},"
            f"train.aud_steps={aud_steps},"
            f"train.batch_size={batch_size},"
            f"model.recon_coeff={recon_coeff},"
            f"model.fair_coeff={fair_coeff},"
            f"optim.learning_rate={learning_rate},"
            f"transfer.n_epochs={transfer_epochs} "
            f"-n new={New},"
            f"index={index} "
            f"--data {data} --dirs local"
        )

      command2 = (
            f"python src/run_unf_clf.py conf/transfer/laftr_then_naive.json "
            f"-o exp_name=\"laftr_example/{exp_name}/Exp_{i}_classification_transfer\","
            f"train.n_epochs={train_epochs},"
            f"train.aud_steps={aud_steps},"
            f"train.batch_size={batch_size},"
            f"model.recon_coeff={recon_coeff},"
            f"model.fair_coeff={fair_coeff},"
            f"optim.learning_rate={learning_rate},"
            f"transfer.n_epochs={transfer_epochs},"
            f"transfer.epoch_number={transfer_epoch_number} "
            f"-n new={New},"
            f"index={index} "
            f"--data {data} --dirs local"
        )


    commands.append((command1, command2))
    return commands

experiments = [
    {'Exp_name': 'Adult', 'data': 'adult', 'num_experiment': 1, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.1, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': False, 'index': 35},
    {'Exp_name': 'Adult', 'data': 'adult', 'num_experiment': 2, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.5, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': False, 'index': 35},
    {'Exp_name': 'Adult', 'data': 'adult', 'num_experiment': 3, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 3, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 1, 'learning_rate': 0.0002, 'transfer_epoch_number': 950, 'New': False, 'index': 35},
    {'Exp_name': 'ACSIncome', 'data': 'ACSIncome', 'num_experiment': 1, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.1, 'learning_rate': 0.001, 'transfer_epoch_number': 950, 'New': False, 'index': 10},
    {'Exp_name': 'ACSIncome', 'data': 'ACSIncome', 'num_experiment': 2, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.5, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': False, 'index': 10},
    {'Exp_name': 'ACSIncome', 'data': 'ACSIncome', 'num_experiment': 3, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 1, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': False, 'index': 10},
    {'Exp_name': 'AdultXc', 'data': 'adult', 'num_experiment': 1, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.1, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': True, 'index': 35},
    {'Exp_name': 'AdultXc', 'data': 'adult', 'num_experiment': 2, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.5, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': True, 'index': 35},
    {'Exp_name': 'AdultXc', 'data': 'adult', 'num_experiment': 3, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 3, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 1, 'learning_rate': 0.0002, 'transfer_epoch_number': 950, 'New': True, 'index': 35},
    {'Exp_name': 'ACSIncomeXc', 'data': 'ACSIncome', 'num_experiment': 1, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.1, 'learning_rate': 0.001, 'transfer_epoch_number': 950, 'New': True, 'index': 10},
    {'Exp_name': 'ACSIncomeXc', 'data': 'ACSIncome', 'num_experiment': 2, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.5, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': True, 'index': 10},
    {'Exp_name': 'ACSIncomeXc', 'data': 'ACSIncome', 'num_experiment': 3, 'train_epochs': 1000, 'transfer_epochs': 500, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 1, 'learning_rate': 0.0005, 'transfer_epoch_number': 950, 'New': True, 'index': 10},
]

#example
experiments =  [
    {'Exp_name': 'Adult', 'data': 'adult', 'num_experiment': 1, 'train_epochs': 100, 'transfer_epochs': 100, 'aud_steps': 2, 'batch_size': 128, 'recon_coeff': 1, 'fair_coeff': 0.1, 'learning_rate': 0.0005, 'transfer_epoch_number': 50, 'New': False, 'index': 35}]

for exp in experiments:
    commands = generate_experiment_commands(
        Exp_name=exp['Exp_name'],
        num_experiments=exp['num_experiment'],
        train_epochs=exp['train_epochs'],
        transfer_epochs=exp['transfer_epochs'],
        aud_steps=exp['aud_steps'],
        batch_size=exp['batch_size'],
        recon_coeff=exp['recon_coeff'],
        fair_coeff=exp['fair_coeff'],
        learning_rate=exp['learning_rate'],
        transfer_epoch_number=exp['transfer_epoch_number'],
        data=exp['data'],
        New=exp['New'],
        index=exp['index']
    )
    for command1, command2 in commands:

        print('#_'+exp['Exp_name']+'_'+str(exp['num_experiment']))
        print(command1)
        print(command2)
        print()
```
