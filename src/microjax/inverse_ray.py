import jax.numpy as jnp
from jax import tree_util
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class CarryData:
    def __init__(self, yi: jnp.int, indx: jnp.ndarray, Nindx: jnp.ndarray, xmin: jnp.ndarray, 
                 xmax: jnp.ndarray, area_x: jnp.ndarray, y: jnp.ndarray, dys: jnp.ndarray, 
                 z_current: jnp.complex128, x0: jnp.float, count_x: jnp.float, count_all: jnp.float, 
                 dz2: jnp.float, dx: jnp.float,  
                 w_center: jnp.complex128, rho2: jnp.float, a: jnp.float, e1: jnp.float,
                 CM2MD: jnp.float, incr: jnp.float, incr_inv: jnp.float, max_iter: int
                 ):
        # final carry
        self.yi = yi
        self.indx = indx
        self.Nindx = Nindx
        self.xmin = xmin
        self.xmax = xmax
        self.area_x = area_x
        self.y = y
        self.dys = dys
        # variables during functions
        self.z_current = z_current
        self.x0 = x0
        self.count_x = count_x
        self.count_all = count_all
        self.dz2 = dz2
        self.dx = dx
        # constants during functions
        self.w_center = w_center
        self.rho2 = rho2
        self.a = a
        self.e1 = e1
        self.CM2MD = CM2MD
        self.incr = incr
        self.incr_inv = incr_inv
        self.max_iter = max_iter

    def tree_flatten(self):
        children = (self.yi, self.indx, self.Nindx, self.xmin, self.xmax, self.area_x, self.y, 
                    self.dys, self.z_current, self.x0, self.count_x, self.count_all, self.dz2, 
                    self.dx, self.w_center, self.rho2, self.a, self.e1, self.CM2MD, self.incr, 
                    self.incr_inv, self.max_iter)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)