import jax
import jax.numpy as jnp
from jax import lax
from flax.linen import Module, compact, Dense, initializers
from typing import Tuple, Callable

class RBFExpansion(Module):
    num_rbf: int
    cutoff: float

    @compact
    def __call__(self, distances):
        centers = jnp.linspace(0, self.cutoff, self.num_rbf)
        distances = jnp.linalg.norm(distances, axis=-1)[:, None]
        centers = centers[None, :]
        return jnp.exp(-0.5 * ((distances - centers) / (self.cutoff / self.num_rbf))**2)

class CosineCutoff(Module):
    cutoff: float

    @compact
    def __call__(self, r):
        return 0.5 * (jnp.cos(jnp.pi * r / self.cutoff) + 1.0) * (r < self.cutoff)

class ShiftedSoftplus(Module):
    @compact
    def __call__(self, x):
        return jnp.log(0.5 * jnp.exp(x) + 0.5)

class CFConv(Module):
    num_features: int
    num_rbf: int
    cutoff: float
    debug: bool = False

    @compact
    def __call__(self, r_ij):
        rbf_expansion = RBFExpansion(self.num_rbf, self.cutoff)
        cutoff_fn = CosineCutoff(self.cutoff)
        activation = ShiftedSoftplus()
        
        if self.debug:   
            print('-------CFCONV LAYER STARTED DEBUG----------')

        rbf_features = rbf_expansion(r_ij)
        if self.debug:
            print(f'RBF_features shape: {rbf_features.shape}')

        x = activation(Dense(384)(rbf_features))
        if self.debug:   
            print(f'RBF_out 1 shape: {x.shape}')

        x = activation(Dense(384)(x))
        if self.debug:
            print(f'RBF_out 2 shape: {x.shape}')

        cutoff = cutoff_fn(jnp.linalg.norm(r_ij, axis=-1))
        x = x * cutoff[:, None]
        
        if self.debug:
            print(f'Final output shape: {x.shape}')
            print('-------CFCONV LAYER ENDED DEBUG----------')

        return x

class Dense(Module):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  use_bias: bool = True

  @compact
  def __call__(self, inputs):
    kernel = self.param(
        'kernel', self.kernel_init, (inputs.shape[-1], self.features)
    )
    y = lax.dot_general(
        inputs,
        kernel,
        (((inputs.ndim - 1,), (0,)), ((), ())),
    )
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      y = y + bias
    return y

class MessageBlock(Module):
    num_features: int
    num_rbf: int
    cutoff: float
    debug: bool = False

    @compact
    def __call__(self, s, v, edge_index, r_ij):
        if self.debug:
            print('-------MESSAGE BLOCK START DEBUG----------')
            print(f'r_ij shape: {r_ij.shape}')
            print(f's shape: {s.shape}')
            print(f'v shape: {v.shape}')
        
        
        r = jnp.linalg.norm(r_ij, axis=-1)
        rbf = jnp.exp(-0.5 * (r[:, None] - jnp.linspace(0, self.cutoff, self.num_rbf))**2 / (self.cutoff / self.num_rbf)**2)
        cutoff = 0.5 * (jnp.cos(jnp.pi * r / self.cutoff) + 1.0) * (r < self.cutoff)

        if self.debug:
            print(f'rbf shape: {rbf.shape}')
            print(f'cutoff shape: {cutoff.shape}')
        
        
        rbf = rbf * cutoff[:, None]

        
        Ws = Dense(self.num_features)(rbf)  # shape: (num_edges, num_features)
        Wvv = Dense(self.num_features * 3)(rbf).reshape(-1, self.num_features, 3)
        Wvs = Dense(self.num_features * 3)(rbf).reshape(-1, self.num_features, 3)
        
        if self.debug:
            print(f'Ws shape: {Ws.shape}')
            print(f'Wvv shape: {Wvv.shape}')
            print(f'Wvs shape: {Wvs.shape}')
        
        
        phi_s = jax.nn.silu(Dense(self.num_features)(s))  # shape: (num_nodes, num_features)
        phi = jax.nn.silu(Dense(3 * self.num_features)(s))  # shape: (num_nodes, 3 * num_features)
        phi_vv, phi_vs, _ = jnp.split(phi, 3, axis=-1)  # split into (num_nodes, num_features)
        
        if self.debug:
            print(f'phi_s shape: {phi_s.shape}')
            print(f'phi_vv shape: {phi_vv.shape}')
            print(f'phi_vs shape: {phi_vs.shape}')
        
        
        
        phi_s_edge = phi_s[edge_index[0]]  # shape: (num_edges, num_features)
        delta_s = jax.ops.segment_sum(
            jnp.einsum('ij,ij->i', phi_s_edge, Ws) * cutoff,  # shape: (num_edges,)
            edge_index[1], num_segments=s.shape[0]
        )

        #
        phi_vv_expanded = phi_vv[edge_index[0]]  # shape: (num_edges, num_features)
        v_selected = v[edge_index[0]]  # shape: (num_edges, num_features, 3)
        delta_v1 = jax.ops.segment_sum(
            jnp.einsum('ijk,ij->ijk', v_selected, phi_vv_expanded) * Wvv,  # shape: (num_edges, num_features, 3)
            edge_index[1], num_segments=v.shape[0]
        )

        # 
        r_ij_norm = r_ij / (jnp.linalg.norm(r_ij, axis=-1, keepdims=True) + 1e-8)
        delta_v2 = jax.ops.segment_sum(
            phi_vs[edge_index[0], :, None] * Wvs * r_ij_norm[:, None, :],  # shape: (num_edges, num_features, 3)
            edge_index[1], num_segments=v.shape[0]
        )

        
        delta_v = delta_v1 + delta_v2
        
        if self.debug:
            print(f'delta_s shape: {delta_s.shape}')
            print(f'delta_v shape: {delta_v.shape}')
            print('-------MESSAGE BLOCK END DEBUG----------')

        return delta_s, delta_v

class UpdateBlock(Module):
    num_features: int
    debug: bool = False

    @compact
    def __call__(self, s, v, delta_s, delta_v):
        if self.debug:
            print('-------UPDATE BLOCK START DEBUG----------')
            print(f's shape: {s.shape}')
            print(f'v shape: {v.shape}')
            print(f'delta_s shape: {delta_s.shape}')
            print(f'delta_v shape: {delta_v.shape}')

        
        U_v = jnp.linalg.norm(v, axis=-1)  # shape: (num_nodes, num_features)
        V_v = jnp.linalg.norm(v, axis=-1)  # shape: (num_nodes, num_features)

        if self.debug:
            print(f'U_v shape: {U_v.shape}')
            print(f'V_v shape: {V_v.shape}')
        
        
        V_v_norm = jnp.linalg.norm(V_v, axis=-1, keepdims=True)  # shape: (num_nodes, 1)
        a_input = jnp.concatenate([s, V_v_norm], axis=-1)  # shape: (num_nodes, num_features + 1)
        
        
        a = jax.nn.silu(Dense(3 * self.num_features)(a_input))  #shape: (num_nodes, 3 * num_features)
        
        
        a_ss, a_sv, a_vv = jnp.split(a, 3, axis=-1)

        if self.debug:
            print(f'a_input shape: {a_input.shape}')
            print(f'a shape: {a.shape}')
            print(f'a_ss shape: {a_ss.shape}')
            print(f'a_sv shape: {a_sv.shape}')
            print(f'a_vv shape: {a_vv.shape}')
        
        
        delta_s_u = a_ss * s + a_sv * jnp.sum(U_v * V_v, axis=-1, keepdims=True)  

        
        delta_v_u = a_vv[:, :, None] * v  # shape: (num_nodes, num_features, 3)
        delta_s = delta_s[:, None]  # shape: (5, 1)

        s_out = s + delta_s_u + delta_s  # shape: (num_nodes, num_features)
        v_out = v + delta_v_u + delta_v  # shape: (num_nodes, num_features, 3)

        if self.debug:
            print(f's_out shape: {s_out.shape}')
            print(f'v_out shape: {v_out.shape}')
            print('-------UPDATE BLOCK END DEBUG----------')
        
        return s_out, v_out

class PaiNN(Module):
    num_features: int
    num_rbf: int = 20
    cutoff: float = 5.0
    debug: bool = False

    @compact
    def __call__(self, s, v, edge_index, r_ij):
        message_block = MessageBlock(self.num_features, self.num_rbf, self.cutoff, self.debug)
        update_block = UpdateBlock(self.num_features, self.debug)

        delta_s, delta_v = message_block(s, v, edge_index, r_ij)
        s_out, v_out = update_block(s, v, delta_s, delta_v)
        return s_out, v_out
