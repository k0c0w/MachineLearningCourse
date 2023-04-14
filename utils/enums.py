from enum import IntEnum

DataProcessTypes = IntEnum('DataProcessTypes', ('standardization', 'normalization'))
SetType = IntEnum('SetType', ('train', 'valid', 'test'))
WeightsInitType = IntEnum('WeightsInitType', ('normal', 'uniform', 'xavier', 'chi'))
GDStoppingCriteria = IntEnum('GDStoppingCriteria', ('epoch', 'gradient_norm', 'difference_norm', 'metric_value'))
