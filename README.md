#  Predicting Patent Success: A Machine Learning Approach to Commercialization Prospects

This the second project for the [Machine learning Course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at EPFL. The project has been done in collaboration with [STIP – Chair of Science Technology and Innovation Policy](https://www.epfl.ch/labs/stip/). 

##  Task description
Understanding the potential commercial success of patents is vital for research institutions, inventors, and industry stakeholders. This documentation delves into the intersection of machine learning techniques and patent commercialization, presenting a tool designed to forecast the likelihood of a patent's success.

In our pursuit of this goal, we provide a comprehensive examination of the underlying framework and share valuable insights into the essential information required for accurate predictions. By leveraging this methodology, we achieved an impressive accuracy of 0.927 and an F1-score of 0.927.

## Data

The authors were not allowed to add the the dataframe to the repository, since it is property of [STIP – Chair of Science Technology and Innovation Policy](https://www.epfl.ch/labs/stip/). In order to obtain the dataframe please contact prakhar.gupta@epfl.ch.
The obtained CSV file needs to be added to the `data` folder under the name of `modelready_220423.csv`. The folder needs to be created in the repository by the user. 

## SetUp

- Download the necessary requirements differentiating on whether the operating system is iOS or Linux/Windows
    - if working on iOS
        ```bash 
        pip install -r requirements_iOS.txt 
        ```
    - if working on Linux/Windows
        ```bash 
        pip install -r requirements.txt 
        ```

- Create the folder named `data` and add the dataframe named `modelready_220423.csv` inside tha folder. 

- To facilitate reproducibility we provide [here](https://drive.google.com/drive/folders/1EkwTVS9IfSViPfPVWtvFV_oOuI9_NL7u?usp=sharing) the parameters of BERT after the finetuning. Please create the folder named `models` and add both trained models to the folder. 

## Folders and Files

- `utilities.py` contains different utils used for the project;
- `requirements.txt` contains the required python packages;
- `requirements_iOS.txt` contains the required python packages for iOS;
- `main.ipynb` main notebook with analysis and results;
- `data_expl_for_ethics.ipynb` notebook used for evaluating possible ethical risks of the project;
- `data` is the folder that needs to be created by the user. Then place inside the the dataframe named `modelready_220423.csv`.
- `models` is the folder that needs to be created by the user. Then place inside the parameters of BERT after the finetuning named `bert_trained_no_ge.pth` and `bert_trained.pth` (these files where downloaded from the provided [link](https://drive.google.com/drive/folders/1EkwTVS9IfSViPfPVWtvFV_oOuI9_NL7u?usp=sharing))

## Authors

- Stefano Viel
- Valerio Ardizio
- Malena Mendilaharzu