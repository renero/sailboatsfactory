from params import Params


class Predict(Params):

    def __init__(self, datasets, oh_encoder):
        super(Predict, self).__init__()
        self._dataset = datasets
        self._oh_encoder = oh_encoder

    def body(self, nn):
        positive_all = 0
        positive_sign = 0
        positive_shape = 0
        num_testcases = self._dataset.X_test.shape[0]
        for j in range((self._dataset.X_test.shape[0]) - 2):
            y = nn.predict(self._dataset.X_test[j:j + 1, :, :])
            y_pred = nn.hardmax(y[0])
            cse_predicted = self._oh_encoder.decode(y_pred)[0]
            cse_actual = self._oh_encoder.decode(
                self._dataset.y_test[j:j + 1, :])[0]
            positive_all += int(cse_actual == cse_predicted)
            positive_sign += int(cse_actual[0] == cse_predicted[0])
            positive_shape += int(cse_actual[-1] == cse_predicted[-1])
            # print('predicted: {} / actual: {}'.format(cse_actual))

        print('Pos.Rate (all/sign/body): {:.3f} / {:.3f} / {:.3f}'.format(
            (positive_all / num_testcases), (positive_sign / num_testcases),
            (positive_shape / num_testcases)))

        return ((positive_all / num_testcases),
                (positive_sign / num_testcases),
                (positive_shape / num_testcases))

    def move(self, nn):
        pos_open = 0
        pos_close = 0
        pos_high = 0
        pos_low = 0
        pred_length = len(self._oh_encoder._states)
        num_predictions = int(self._dataset.y_test.shape[1] / pred_length)

        for j in range((self._dataset.X_test.shape[0]) - 2):
            y = nn.predict(self._dataset.X_test[j:j + 1, :, :])
            Y_pred = [
                nn.hardmax(
                    y[0][i * pred_length:(i * pred_length) + pred_length - 1])
                for i in range(num_predictions)
            ]
            move_predicted = [
                self._oh_encoder.decode(Y_pred[i])[0]
                for i in range(num_predictions)
            ]
            move_actual = [
                self._oh_encoder.decode(
                    self._dataset.y_test[j:j + 1, :])[0]
                for i in range(num_predictions)
            ]
            pos_open += int(move_actual[0] == move_predicted[0])
            pos_high += int(move_actual[1] == move_predicted[1])
            pos_low += int(move_actual[2] == move_predicted[2])
            pos_close += int(move_actual[3] == move_predicted[3])

        num_testcases = (self._dataset.X_test.shape[0]) - 2
        print('Pos.Rate (O/H/L/C): {:.4f} : {:.4f} : {:.4f} : {:.4f} ~ {:.4f}'.
              format((pos_open / num_testcases), (pos_high / num_testcases),
                     (pos_low / num_testcases), (pos_close / num_testcases),
                     ((pos_open + pos_high + pos_low + pos_close) /
                      (num_testcases * 4))))
