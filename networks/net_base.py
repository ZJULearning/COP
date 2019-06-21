from abc import ABC, abstractmethod

class NetBase(ABC):
    @abstractmethod
    def get_params_and_calculation_from_channel_num():
        pass

    @abstractmethod
    def get_weights_from_model():
        pass

    @abstractmethod
    def restore_weights():
        pass

    @abstractmethod
    def network():
        pass

    @abstractmethod
    def loss():
        pass

    @abstractmethod
    def metric_op():
        pass