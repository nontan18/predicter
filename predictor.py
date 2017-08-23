import chainer
from chainer import functions as F
from chainer import links as L
class PredictModel(chainer.Chain):
    """docstring for PredictModel."""
    # V1
    # def __init__(self, input_num):
    #     self.input_num = input_num
    #     super(PredictModel, self).__init__(
    #         fc1=L.Linear(self.input_num,1024),
    #         lstm1=L.LSTM(1024,1024),
    #         fc2=L.Linear(1024,1024),
    #         fc3=L.Linear(1024,3),
    #     )

    # V2
    def __init__(self, input_num):
        self.input_num = input_num
        super(PredictModel, self).__init__(
            fc1=L.Linear(self.input_num,8),
            lstm1=L.LSTM(8,8),
            fc2=L.Linear(128,128),
            fc3=L.Linear(8,3),
        )

    def predict(self, x):
        x = chainer.Variable(x.astype(np.float32))
        return self.__call__(x)

    def __call__(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.lstm1(h)
        h = F.relu(h)
        # h = self.fc2(h)
        h = self.fc3(h)
        return F.softmax(h)
