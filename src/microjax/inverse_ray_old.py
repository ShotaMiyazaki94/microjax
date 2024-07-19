import jax.numpy as jnp
from .poly_solver import poly_roots
from .point_source import lens_eq
import jax
from jax.tree_util import register_pytree_node_class

def source_profile_limb1(dz, u1=0.0):
    mu = jnp.sqrt(1.0 - dz*dz)
    return 1 - u1 * (1.0 - mu)

@register_pytree_node_class
class CarryData:
    def __init__(self, yi, z_current, dx, count_x, count_all, xmax, xmin, 
                 area_x, y, dys, indx, Nindx, x0, w_center, rho2, CM2MD, 
                 a, e1, incr, incr_inv, max_iter, dz2):
        self.yi = yi
        self.z_current = z_current
        self.dx = dx
        self.count_x = count_x
        self.count_all = count_all
        self.xmax = xmax
        self.xmin = xmin
        self.area_x = area_x
        self.y = y
        self.dys = dys
        self.indx = indx
        self.Nindx = Nindx
        self.x0 = x0
        self.w_center = w_center
        self.rho2 = rho2
        self.CM2MD = CM2MD
        self.a = a
        self.e1 = e1
        self.incr = incr
        self.incr_inv = incr_inv
        self.max_iter = max_iter
        self.dz2 = dz2

    def tree_flatten(self):
        children = (self.yi, self.z_current, self.dx, self.count_x, self.count_all, 
                    self.xmax, self.xmin, self.area_x, self.y, self.dys, self.indx,
                    self.Nindx, self.x0, self.w_center, self.rho2, self.CM2MD, 
                    self.a, self.e1, self.incr, self.incr_inv, self.max_iter, self.dz2)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def image_area0(w_center, rho, z_init, dy, carry, nlenses=2, **_params):
    """
    """
    q, s = _params["q"], _params["s"]
    yi, indx, Nindx, xmax, xmin, area_x, y, dys = carry 
    max_iter = len(dys) // 2
    CM2MD = -0.5 * s * (1 - q)/(1 + q) 
    z_current = z_init
    x0 = z_init.real
    a = 0.5 * s
    e1 = q / (1.0 + q) 
    dz2 = 9999.9999
    incr = jnp.abs(dy)
    incr_inv = 1.0 / incr
    dx = incr 
    count_x = 0.0
    count_all = 0.0
    rho2 = rho * rho

    carry_init = (yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys,
                  indx, Nindx, x0, w_center, rho2, CM2MD, a, e1, incr, incr_inv, max_iter, dz2)
    result = jax.lax.while_loop(cond_fn, update_fn, carry_init)
    
    count_all, carry = result[4], result[:8]
    return count_all, carry

def cond_fn(carry):
    yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys,\
        indx, Nindx, x0, w_center, rho2, CM2MD, a, e1, incr, incr_inv, max_iter, dz2 = carry
    return (yi < len(xmin)) & (z_current.imag <= (max_iter / incr_inv))

def update_fn(carry):
    yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys,\
        indx, Nindx, x0, w_center, rho2, CM2MD, a, e1, incr, incr_inv, max_iter, dz2_last = carry
    
    z_current_mid = z_current + CM2MD
    zis_mid = lens_eq(z_current_mid, a=a, e1=e1)
    zis = zis_mid - CM2MD
    dz = jnp.abs(w_center - zis)
    dz2 = dz ** 2

    yi, z_current, dx, count_x, count_all, xmax, xmin, area_x = jax.lax.cond(
        dz2 <= rho2,
        update_inside_source,
        update_outside_source,
        (yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys, \
         indx, Nindx, x0, incr, dz2_last, rho2, incr_inv, max_iter, dz)
    )
    
    z_current += dx
    return yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys, indx, Nindx, x0, w_center, rho2, CM2MD, a, e1, incr, incr_inv, max_iter, dz2

def update_inside_source(args):
    yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys,\
        indx, Nindx, x0, dz, rho2, incr, dz2_last, dy = args
    count_eff = source_profile_limb1(dz)
    count_x += float(count_eff)
    xmax = jax.lax.cond((dx == -incr) & (count_x == 0.0),
                        lambda _: xmax.at[yi].set(z_current.real - dx),
                        lambda _: xmax)
    return yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys, indx, Nindx, x0, dz2_last, dy

def update_outside_source(args):
    yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys,\
        indx, Nindx, x0, dz, rho2, incr, dz2_last, dy, incr_inv, max_iter = args
    
    def positive_run_fn(args):
        yi, z_current, dx, xmax, xmin = args
        xmax = jax.lax.cond(dz2_last <= rho2,
                            lambda _: xmax.at[yi].set(z_current.real),
                            lambda _: xmax)
        dx = -incr
        z_current = jnp.complex128(x0 + 1j * z_current.imag)
        xmin = xmin.at[yi].set(z_current.real + dx)
        return yi, z_current, dx, xmax, xmin
    
    def negative_run_fn(args):
        yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys, indx, Nindx, incr, x0, dy, incr_inv, max_iter = args
        xmin = jax.lax.cond(dz2_last <= rho2,
                            lambda _: xmin.at[yi].set(z_current.real),
                            lambda _: xmin)
        
        def inner_cond_fn(_):
            return (z_current.real >= xmin[yi-1] + incr) & (yi != 0) & (count_x == 0)

        def inner_true_fn(args):
            yi, z_current, dx = args
            z_current += dx
            return yi, z_current, dx
        
        yi, z_current, dx = jax.lax.cond(inner_cond_fn(None),
                                         inner_true_fn,
                                         lambda args: args,
                                         (yi, z_current, dx))
        count_all += count_x
        area_x = area_x.at[yi].set(count_x)
        y = y.at[yi].set(z_current.imag)
        dys = dys.at[yi].set(dy)
        dys = jax.lax.cond(count_x == 0.0,
                           lambda _: dys.at[yi].set(-dy),
                           lambda _: dys)
        
        def already_counted_fn(args):
            yi, indx, Nindx, xmax, xmin, area_x, y, dys = args
            area_x = area_x.at[yi].set(0.0)
            return yi, indx, Nindx, xmax, xmin, area_x, y, dys

        def not_counted_fn_inner_loop(args, j):
            yi, indx, Nindx, xmax, xmin, area_x, y, dys, incr_inv, max_iter, z_current, count_x = args
            ind = indx[int(z_current.imag * incr_inv + max_iter)][j]
            already_counted = (xmin[yi] < xmax[ind]) & (xmax[yi] > xmin[ind])
            args = jax.lax.cond(already_counted, 
                                lambda args: already_counted_fn((yi, indx, Nindx, xmax, xmin, area_x, y, dys)),
                                lambda args: args, 
                                args)
            return args, None

        def not_counted_fn(args):
            yi, indx, Nindx, incr_inv, max_iter, z_current, count_x = args
            y_index = int(z_current.imag * incr_inv + max_iter)
            args = (yi, indx, Nindx, xmax, xmin, area_x, y, dys, incr_inv, max_iter, z_current, count_x)
            args, _ = jax.lax.scan(not_counted_fn_inner_loop, args, jnp.arange(Nindx[y_index]))
            yi, indx, Nindx, xmax, xmin, area_x, y, dys, incr_inv, max_iter, z_current, count_x = args
            indx = indx.at[y_index, Nindx[y_index]].set(yi)
            Nindx = Nindx.at[y_index].add(1)
            yi += 1
            dx = incr
            x0 = xmax[yi-1]
            z_current = jnp.complex128(x0 + 1j * (z_current.imag + dy))
            count_x = 0.0
            return yi, indx, Nindx, xmax, xmin, area_x, y, dys, dx, x0, z_current, count_x

        yi, indx, Nindx, xmax, xmin, area_x, y, dys, dx, x0, z_current, count_x = jax.lax.cond(inner_cond_fn(None),
            already_counted_fn,
            not_counted_fn,
            (yi, indx, Nindx, incr_inv, max_iter, z_current, count_x))
        
        return yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys, indx, Nindx, x0

    return jax.lax.cond(dx == incr,
                        positive_run_fn,
                        negative_run_fn,
                        (yi, z_current, dx, count_x, count_all, xmax, xmin, area_x, y, dys, indx, Nindx, x0, dz2_last, dy, incr_inv, max_iter))