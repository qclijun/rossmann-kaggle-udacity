import sys

import numpy as np

from ross_util import write_submission, submit_to_kaggle
from ross_model import NN_Embedding
from ross_basemodel import NN_Embedding_Base
from ross_data import get_dataset, get_dataset_5_9

MODEL = NN_Embedding
#MODEL = NN_Embedding_Base

N_NETWORKS = 1
VALIDATION_WEEKS = 0
DO_EVAL = VALIDATION_WEEKS > 0

INIT_EPOCH = 0
EPOCHS = 25
BATCH_SIZE = 2048

FILT_STORES_FOR_TRAIN = True
FILT_STORES_FOR_VALID = True
SAVE_CHECKPT = False
SAVE_MODEL = True
PRINT_SUMMARY = True
SUBMIT = True

saved_model_file = './models/model_1.hdf5'
#saved_model_file = './output/checkpt/weights.40.hdf5'
submission_file = './output/pred2018-3-29.csv'
model_dir = './models/'

def load_models_from_hdf(model_files):
    models = []
    for model_file in model_files:
        print('loading model from', model_file)
        model = MODEL()
        model.model.load_weights(model_file)
        models.append(model)
    return models


def main():
    train_set, valid_set, X_test = get_dataset(VALIDATION_WEEKS, FILT_STORES_FOR_TRAIN, FILT_STORES_FOR_VALID)
    models = []
    for i in range(N_NETWORKS):
        print('\nNN_Embedding Model', i+1)
        print('-'*50)
        model = MODEL(print_model_summary=PRINT_SUMMARY, save_checkpt=SAVE_CHECKPT)

        if INIT_EPOCH > 0:
            model.model.load_weights(saved_model_file)
        model.fit(train_set, valid_set, batch_size=BATCH_SIZE, epochs=EPOCHS + INIT_EPOCH, init_epoch=INIT_EPOCH)

        if SAVE_MODEL:
            print('saving model', i+1)
            model.model.save(model_dir + 'model_'+str(i+1)+'.hdf5')

        models.append(model)

        # models.extend(load_models_from_hdf(['./output/checkpt/weights.{:02d}.hdf5'.format(i)
        #                                     for i in range(EPOCHS-2, EPOCHS)]))

    if DO_EVAL:
        errors = [model.eval() for model in models]
        for i,err in enumerate(errors):
            print('Model {}: RMSPE = {}'.format(i+1, err))
        print('Mean:', np.mean(errors))
        for i, model in enumerate(models):
            model.plot_loss('./output/loss_model_{}.png'.format(i+1))

    if SUBMIT:
        if len(sys.argv) > 1:
            filename = './output/'+sys.argv[1]
        else:
            filename = submission_file
        write_submission(models, X_test, filename)
        submit_to_kaggle(filename)

        if N_NETWORKS > 1:
            for i in range(N_NETWORKS):
                filename = '{}_model_{}.csv'.format(submission_file[:-4], i+1)
                write_submission([models[i]], X_test, filename)
                submit_to_kaggle(filename)

if __name__=='__main__':
    main()







