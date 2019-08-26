import csv
import numpy as np

def store_training_loss(history, filepath):

    loss = history.history['loss']
    # mse_loss = history.history['mse']
    # nlml_loss = history.history['nlml']
    # gp_1_mse = history.history['gp_1_mse']
    # gp_1_nlml = history.history['gp_1_nlml']
    # val_nlml = history.history['val_nlml']
    # val_mse = history.history['val_mse']
    epoches = int(history.params['epochs'])
    if epoches > len(loss):
        epoches = len(loss)

    with open(filepath, 'w', newline='') as writer:
        csv_writer = csv.writer(writer, delimiter=',')
        csv_writer.writerow(history.history.keys())
        for i in range(epoches):
            data_row = []
            for loss_metric, values in history.history.items():
                data_row = data_row + [values[i]]

            csv_writer.writerow(data_row)


def flatten_test_predict(y_tests, y_predicts):
    if len(y_tests) == 1:
        y_tests = y_tests[0]

    if len(y_predicts) == 1:
        y_predicts = y_predicts[0]

    if not isinstance(y_tests, np.ndarray) or not isinstance(y_predicts, np.ndarray):
        raise Exception('bad input data for y_tests or y_predicts')

    y_tests = y_tests.ravel()
    y_predicts = y_predicts.ravel()

    return y_tests, y_predicts


def store_predict_points(y_tests, y_predicts, filepath):
    n = len(y_tests)
    if n != len(y_predicts):
        raise Exception('bad testing samples and predictions')

    with open(filepath, 'w', newline='') as writer:
        csv_writer = csv.writer(writer, delimiter=',')
        csv_writer.writerow(["y_test", "y_pred"])

        for i in range(len(y_tests)):
            csv_writer.writerow([y_tests[i], y_predicts[i]])

        print("Done writing prediction result to", filepath)