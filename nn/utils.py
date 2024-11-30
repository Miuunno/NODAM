import jax

class CFG:
    KEY = None

def set_key(key):
    CFG.KEY = jax.random.key(key)

def normal(*shape):
    return jax.random.normal(key=CFG.KEY, shape=shape)

def shuffle(x):
    return jax.random.permutation(key=CFG.KEY, x=x)

def choice(x, *shape):
    return jax.random.choice(key=CFG.KEY, a=x, shape=shape)