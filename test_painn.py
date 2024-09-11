import jax
import jax.numpy as jnp
from blocks import PaiNN


def loss_fn(params, s, v, edge_index, r_ij, s_target, v_target):

    s_out, v_out = model.apply(params, s, v, edge_index, r_ij)
    

    s_loss = jnp.mean((s_out - s_target) ** 2)
    v_loss = jnp.mean((v_out - v_target) ** 2)
    

    return s_loss + v_loss


num_features = 64
num_rbf = 20
cutoff = 5.0

model = PaiNN(num_features=num_features, num_rbf=num_rbf, cutoff=cutoff, debug=True)


key = jax.random.PRNGKey(0)
s = jax.random.normal(key, (5, num_features))
v = jax.random.normal(key, (5, num_features, 3))
edge_index = jax.random.randint(key, (2, 10), 0, 5)
r_ij = jax.random.normal(key, (10, 3))


s_target = jax.random.normal(key, (5, num_features))
v_target = jax.random.normal(key, (5, num_features, 3))


params = model.init(key, s, v, edge_index, r_ij)


grad_fn = jax.grad(loss_fn)

grads = grad_fn(params, s, v, edge_index, r_ij, s_target, v_target)


print("Gradients:", grads)


