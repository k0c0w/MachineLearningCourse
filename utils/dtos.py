class ModelDescription:
    def __init__(self, m, l, valid_error=None, test_error=None):
        self.M = m
        self.Lambda = l
        self.ErrorOnValid = valid_error
        self.ErrorOnTest = test_error

    def __str__(self):
        valid_error = f' validation_error={self.ErrorOnValid}' if self.ErrorOnValid else ''
        test_error = f' test_error={self.ErrorOnTest}' if self.ErrorOnTest else ''
        return f'M={self.M} Lambda={self.Lambda}{valid_error}{test_error}'
