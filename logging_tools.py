class DummyLogger:
    """ Dummy logger used when not using WandB; mimics the structure expected by the main logging functions. """
    def __init__(self):
        self.experiment = self.DummyExperiment()

    def log_scalar(self, name, value, step): pass

    def log(self, value, commit): pass
    
    class DummyExperiment:
        def log(self, *args, **kwargs): pass