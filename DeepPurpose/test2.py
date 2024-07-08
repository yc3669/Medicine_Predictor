from DeepPurpose import DDI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

# load DB Binary Data
X_drugs, y = load_Xiangyang_data()
train = X_drugs[:50000]
test = X_drugs[50000:65000]
val = X_drugs[65000:]

drug_encoding = 'CNN'
train = data_process(X_drug=train, X_drug_=train, y=y[:50000],
                                           drug_encoding=drug_encoding,
                                           split_method='no_split',
                                           frac=[0.4, 0.4, 0.2],
                                           random_seed = 1)
test = data_process(X_drug=test, X_drug_=test, y=y[50000:65000],
                                           drug_encoding=drug_encoding,
                                           split_method='no_split',
                                           frac=[0.4, 0.4, 0.2],
                                           random_seed = 1)
val = data_process(X_drug=val, X_drug_=val, y=y[65000:],
                                           drug_encoding=drug_encoding,
                                           split_method='no_split',
                                           frac=[0.4, 0.4, 0.2],
                                           random_seed = 1)

config = generate_config(drug_encoding = drug_encoding,
                         cls_hidden_dims = [512],
                         train_epoch = 20,
                         LR = 0.001,
                         batch_size = 128,
                        )

model = models.model_initialize(**config)
model.train(train, val, test)
