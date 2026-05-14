from abc import ABC, abstractmethod

class BaseResidualModel(ABC):
    registry: dict[str, type["BaseResidualModel"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not BaseResidualModel:
            BaseResidualModel.registry[cls.key()] = cls
    
    @classmethod
    @abstractmethod
    def key(cls) -> str:
        pass

    @classmethod
    def create(cls, key: str, **kwargs):
        model_cls = cls.registry.get(key)
        if model_cls is None:
            available = ", ".join(sorted(cls.registry)) or "none"
            raise ValueError(f"Unknown residual model '{key}'. Available: {available}")
        return model_cls(**kwargs)
