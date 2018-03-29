
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ross_util  import write_submission, submit_to_kaggle
from ross_main import load_models_from_hdf
from ross_data import load_data

train, test = load_data()
X_test = test
#model_files = ['./output/checkpt/weights.12.hdf5']
#model_files = ['./output/checkpt/weights.{:02d}.hdf5'.format(i) for i in range(20, 25)]
model_files = ['./models/model_{}.hdf5'.format(i+1) for i in range(10)]
submission_file = './output/submission.10models.csv'

#model_files = ['./models/model_{}.hdf5'.format(i+1)  for i in range(10)]

models = load_models_from_hdf(model_files)
write_submission(models, X_test, submission_file)
submit_to_kaggle(submission_file, message='')
# for i, model in enumerate(models):
#     filename = './output/pred_model_{}.csv'.format(i+1)
#     write_submission([model], X_test, filename)
#     submit_to_kaggle(filename, message='')

# from keras.utils import plot_model
# plot_model(models[0].model, to_file='score294_model.png', show_shapes=True)