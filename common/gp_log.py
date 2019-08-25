import csv


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
        csv_writer.writerow(history.keys())
        for i in range(epoches):
            data_row = []
            for loss_metric, values in history.items():
                data_row = data_row + [values[i]]

            csv_writer.writerow(data_row)


def store_predict_points(y_tests, y_predicts, filepath):
    n = len(y_tests)
    if n != len(y_predicts):
        raise Exception('bad testing samples and predictions')

    with open(filepath, 'w', newline='') as writer:
        csv_writer = csv.writer(writer, delimiter=',')
        csv_writer.writerow(["y_test", "y_pred"])

        for i in range(n):
            csv_writer.writerow([y_tests[i], y_predicts[i]])

        print("Done writing prediction result to", filepath)