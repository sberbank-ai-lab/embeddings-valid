def create(type, params):
    i = type.split('.')
    mod = __import__('.'.join(i[:-1]), fromlist=[i[-1]])
    cls = getattr(mod, i[-1])
    if cls is None:
        raise AttributeError(f'Unknown model type: "{type}"')

    return cls(**params)
