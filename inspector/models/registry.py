MODEL_REGISTRY = {}


def register_model(name, dataset_type=None, task_type=None):
    def decorator(cls):
        MODEL_REGISTRY[(name, dataset_type, task_type)] = cls
        return cls

    return decorator


def _lookup(name, dataset_type, task_type):
    for key in [
        (name, dataset_type, task_type),
        (name, dataset_type, None),
        (name, None, task_type),
        (name, None, None),
    ]:
        if key in MODEL_REGISTRY:
            return MODEL_REGISTRY[key]
    raise NotImplementedError(
        f"No model registered for name='{name}', "
        f"dataset_type='{dataset_type}', task_type='{task_type}'.\n"
        f"Registered keys: {list(MODEL_REGISTRY.keys())}"
    )


def get_model_from_registry(cfg, in_dim, out_dim):
    cls = _lookup(
        cfg.model.name, cfg.dataset.type, cfg.dataset.task_type
    )
    return cls.from_cfg(cfg, in_dim, out_dim)


def list_models():
    return list(MODEL_REGISTRY.keys())


def delete_model(model_key_entry):
    res = MODEL_REGISTRY.pop(model_key_entry, None)
    if res is None:
        print(f"{model_key_entry} not in model registry.")
        return False
    return True
