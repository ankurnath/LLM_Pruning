## Setup

The included [``environment.yml``](environment.yml) file will produce a working environment called ``pruning``.



    >>> cd LLM_Pruning
    >>> conda env create -f environment.yml 
    >>> conda activate eoh


## Data Download

To download datasets, run the following command. Replace Facebook and Wiki with your desired dataset names.

    >>> python data_processing/data_download.py --datasets Facebook Wiki

To split the dataset into training and test sets, specify the dataset and the desired split ratio (e.g., 30% train data):

    >>> python data_processing/train_test_split.py --dataset Facebook --ratio 0.3 


## Remark 

For the implementation details of COMBHelper, LeNSE, GCOMB-P and QuickPrune, please refer to the original repository.


## Feature Space Search

    >>> python main.py --problem "Maximum Coverage" --budget 100 --dataset "HK" --depth 5 --beam_size 3 --expansion_factor 2

## Finetune
    >>> python finetune.py --problem "Maximum Coverage" --budget 100 --base_dataset "HK" --dataset "Facebook"

## Test
    >>> python test.py --problem "Maximum Coverage" --budget 100 --base_dataset "HK" --train_dataset "Facebook" --test_dataset "Facebook"





