import csv


def store_training_loss(history, filepath):

    loss = history.history['loss']
    mse_loss = history.history['mse']
    nlml_loss = history.history['nlml']
    gp_1_mse = history.history['gp_1_mse']
    gp_1_nlml = history.history['gp_1_nlml']

    epoches = int(history.params['epochs'])
    if epoches > len(loss):
        epoches = len(loss)

    with open(filepath, 'w', newline='') as writer:
        csv_writer = csv.writer(writer, delimiter=',')
        csv_writer.writerow(["epoch", "mse", "nlml", "gp_1_mse", "gp_1_nlml", "loss"])
        for i in range(epoches):
            mse = mse_loss[i]
            nlml = nlml_loss[i]
            gp1_mse = gp_1_mse[i]
            gp1_nlml = gp_1_nlml[i]
            l = loss[i]
            csv_writer.writerow([i, mse, nlml, gp1_mse, gp1_nlml, l])


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