from kgp.models import Model as GPModel
from keras.engine.training import _standardize_input_data


class CustomGPModel(GPModel):

    def __init__(self, inputs, outputs, name=None):
        super(GPModel, self).__init__(inputs, outputs, name)


    def predict(self, X, X_tr=None, Y_tr=None,
                batch_size=32, return_var=False, verbose=0):
        """Generate output predictions for the input samples batch by batch.

        Arguments:
        ----------
            X : np.ndarray or list of np.ndarrays
            batch_size : uint (default: 128)
            return_var : bool (default: False)
                Whether predictive variance is returned.
            verbose : uint (default: 0)
                Verbosity mode, 0 or 1.

        Returns:
        --------
            preds : a list or a tuple of lists
                Lists of output predictions and variance estimates.
        """
        # Update GP data if provided (and grid if necessary)
        if X_tr is not None and Y_tr is not None:
            X_tr, Y_tr, _ = self._standardize_user_data(
                X_tr, Y_tr,
                sample_weight=None,
                class_weight=None,
                check_batch_axis=False,
                batch_size=batch_size)
            H_tr = self.transform(X_tr, batch_size=batch_size)
            for gp, h, y in zip(self.output_gp_layers, H_tr, Y_tr):
                gp.backend.update_data('tr', h, y)
                if gp.update_grid:
                    gp.backend.update_grid('tr')

        # Validate user data
        X = _standardize_input_data(
            X, self._feed_input_names, self._feed_input_shapes,
            check_batch_axis=False, exception_prefix='input')

        H = self.transform(X, batch_size=batch_size)

        preds = []
        for l, h in zip(self.output_layers, H):

            if l.name.startswith('gp'):
                preds.append(l.backend.predict(h, return_var=return_var))
            else:
                preds.append(l.backend.predict(h, return_var=return_var))


        if return_var:
            preds = map(list, zip(*preds))

        return preds