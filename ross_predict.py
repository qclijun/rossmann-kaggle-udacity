
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .ross_model import  NN_Embedding
from .ross_main import write_submission, load_data


model_dir = './output/a'


def load_models_from_hdf(model_files):
    models = []
    for model_file in model_files:
        model = NN_Embedding()
        model.model.load_weights(model_file)
        models.append(model)
    return models


train, test = load_data()
X_test = test
model_files = ['./output/checkpt2/weights.18-0.0112.hdf5']
#model_files = ['./solution2nd/models_'+str(i+1) +'.hdf5' for i in range(10)]
#model_files = ['./models_{}.hdf5'.format(i+1) for i in range(20)]
models = load_models_from_hdf(model_files)
write_submission(models, X_test, './output/ross_18.csv')
# for i, model in enumerate(models):
#     write_submission([model], X_test, './output/ross_15_{}.csv'.format(i+1))
