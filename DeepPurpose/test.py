from DeepPurpose import CompoundPred
from DeepPurpose import DDI
from DeepPurpose import MultiModel
from DeepPurpose.utils import *
from DeepPurpose.dataset import *


SAVE_PATH='./saved_path'
import os
if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)


def data_process_multimodel(X_drug=None, X_target=None, y=None, drug_encoding=None, target_encoding=None,
                 split_method='random', frac=[0.7, 0.1, 0.2], random_seed=1, sample_frac=1, mode='DTI', X_drug_=None,
                 X_target_=None):
    if random_seed == 'TDC':
        random_seed = 1234

    if split_method == 'repurposing_VS':
        y = [-1] * len(X_drug)  # create temp y for compatitibility

    model_number = len(drug_encoding)

    if model_number == 1:
        print('1 Model')
        df_data = pd.DataFrame(zip(X_drug, y))
        df_data.rename(columns={0: 'SMILES',
                                1: 'Label'},
                       inplace=True)
        print('in total: ' + str(len(df_data)) + ' drugs')

    elif model_number == 2:
        print('2 Models')

        df_data = pd.DataFrame(zip(X_drug, X_drug, y))
        df_data.rename(columns={0: 'SMILES 1',
                                1: 'SMILES 2',
                                2: 'Label'},
                       inplace=True)
        print('in total: ' + str(len(df_data)) + ' drug-drug pairs')

    elif model_number == 3:
        print('3 Models')

        df_data = pd.DataFrame(zip(X_drug, X_drug, X_drug, y))
        df_data.rename(columns={0: 'SMILES 1',
                                1: 'SMILES 2',
                                2: 'SMILES 3',
                                3: 'Label'},
                       inplace=True)
        print('in total: ' + str(len(df_data)) + ' drug-drug pairs')

    if sample_frac != 1:
        df_data = df_data.sample(frac=sample_frac).reset_index(drop=True)
        print('after subsample: ' + str(len(df_data)) + ' data points...')

    if model_number == 1:
        df_data = encode_drug(df_data, drug_encoding[0])
    elif model_number == 2:
        df_data = encode_drug(df_data, drug_encoding[0], 'SMILES 1', 'drug_encoding_1')
        df_data = encode_drug(df_data, drug_encoding[1], 'SMILES 2', 'drug_encoding_2')
    elif model_number == 3:
        df_data = encode_drug(df_data, drug_encoding[0], 'SMILES 1', 'drug_encoding_1')
        df_data = encode_drug(df_data, drug_encoding[1], 'SMILES 2', 'drug_encoding_2')
        df_data = encode_drug(df_data, drug_encoding[2], 'SMILES 3', 'drug_encoding_3')

    # dti split
    if model_number > 1:
        if split_method == 'random':
            train, val, test = create_fold(df_data, random_seed, frac)
        elif split_method == 'no_split':
            return df_data.reset_index(drop=True)
    else:
        # drug property predictions
        if split_method == 'repurposing_VS':
            train = df_data
            val = df_data
            test = df_data
        elif split_method == 'no_split':
            print('do not do train/test split on the data for already splitted data')
            return df_data.reset_index(drop=True)
        else:
            train, val, test = create_fold(df_data, random_seed, frac)

    print('Done.')
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# X_drugs, _, y = load_AID1706_SARS_CoV_3CL()
X_drugs, y = load_Xiangyang_data()
train = X_drugs[:50000]
test = X_drugs[50000:65000]
val = X_drugs[65000:]


drug_encoding = ['CNN','MPNN']
# drug_encoding = 'CNN'
train = data_process_multimodel(X_drug=train, y=y[:50000],
                                           drug_encoding=drug_encoding,
                                           split_method='no_split',
                                           frac=[0.4, 0.4, 0.2],
                                           random_seed = 1)
test = data_process_multimodel(X_drug=test, y=y[50000:65000],
                                           drug_encoding=drug_encoding,
                                           split_method='no_split',
                                           frac=[0.4, 0.4, 0.2],
                                           random_seed = 1)
val = data_process_multimodel(X_drug=val, y=y[65000:],
                                           drug_encoding=drug_encoding,
                                           split_method='no_split',
                                           frac=[0.4, 0.4, 0.2],
                                           random_seed = 1)
# train, test, val = data_process(X_drug=X_drugs, y=y,
#                                            drug_encoding=drug_encoding,
#                                            split_method='random',
#                                            frac=[0.7, 0.1, 0.2],
#                                            random_seed = 1)
config = generate_config(drug_encoding=drug_encoding[0],
                             cls_hidden_dims=[512],
                             train_epoch=20,
                             LR=0.001,
                             batch_size=128,
                            )
config_list = []
if len(drug_encoding) == 1:
    model = CompoundPred.model_initialize(**config)
else:
    for i in range(len(drug_encoding)):
        config = generate_config(drug_encoding=drug_encoding[i],
                                 cls_hidden_dims=[512],
                                 train_epoch=20,
                                 LR=0.001,
                                 batch_size=128,
                                )
        config_list.append(config)
    model = MultiModel.model_initialize(config_list)
model.train(train, val, test)
#
# X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
#
# _ = models.repurpose(X_repurpose, model, drug_name)