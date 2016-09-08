from .model import Model

class TestModel(Model):

    def __init__(self, sess, config):
        super(TestModel, self).__init__(sess, config)
        print("test_model initialize")
        self.build_model()

    def build_model(self):
        print("batch size is", self.config.batch_size)
        print("[test] build model")
