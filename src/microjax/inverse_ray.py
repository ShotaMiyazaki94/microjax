import jax.numpy as jnp
from jax import tree_util
from jax.tree_util import register_pytree_node_class
import jax
from jax import lax
from .point_source import lens_eq, _images_point_source

def source_profile_limb1(dz2, u1=0.0):
    mu = jnp.sqrt(1.0 - dz2)
    return 1 - u1 * (1.0 - mu)

def image_area0(w_center, rho, z_init, dy, carry, nlenses=2, **_params):
    q, s = _params["q"], _params["s"]
    yi, indx, Nindx, xmax, xmin, area_x, y, dys = carry 
    max_iter = len(dys) // 2
    CM2MD = -0.5 * s * (1 - q)/(1 + q) 
    z_current = z_init
    x0 = z_init.real
    a = 0.5 * s
    e1 = q / (1.0 + q) 
    dz2 = jnp.inf
    incr = jnp.abs(dy)
    incr_inv = 1.0 / incr
    dx = incr 
    count_x = 0.0
    count_all = 0.0
    rho2 = rho * rho
    finish = jnp.bool_(False)

    carry_init = CarryData(yi=yi, indx=indx, Nindx=Nindx, xmin=xmin, 
                            xmax=xmax, area_x=area_x, y=y, dys=dys,
                            z_current=z_current, x0=x0, count_x=count_x, 
                            count_all=count_all, dz2=dz2, dz2_last=dz2, dx=dx, finish=finish,
                            w_center=w_center, rho2=rho2, a=a, e1=e1, CM2MD=CM2MD, 
                            incr=incr, incr_inv=incr_inv, max_iter=max_iter, nlenses=nlenses)   
    
    result = lax.while_loop(cond_fun = finish_area0, 
                            body_fun = update_dy, 
                            init_val = carry_init)

    carry_return = (result.yi, result.indx, result.Nindx, result.xmin, 
                    result.xmax, result.area_x, result.y, result.dys)
    
    return result.count_all, carry_return  

def finish_area0(carry):
    return ~carry.finish

def update_dy(carry):
    z_current_mid = carry.z_current + carry.CM2MD
    zis_mid = lens_eq(z_current_mid, a=carry.a, e1=carry.e1, nlenses=carry.nlenses)
    zis = zis_mid - carry.CM2MD
    carry.dz2_last = carry.dz2
    dz = jnp.abs(carry.w_center - zis)
    carry.dz2 = dz**2

    cond_inside = carry.dz2 <= carry.rho2 
    carry = jax.lax.cond(cond_inside,
                         update_inside_source,
                         update_outside_source,
                         carry)
    
    carry.z_current = lax.cond(carry.finish,
                               lambda _: carry.z_current,
                               lambda _: carry.z_current + carry.dx,
                               None)
    return carry

def update_inside_source(carry):
    count_eff = source_profile_limb1(carry.dz2)
    carry.count_x += count_eff.astype(float)
    carry.xmax = lax.cond((carry.dx == -carry.incr) & (carry.count_x==0.0), 
                          lambda _: carry.xmax[carry.yi].set(carry.z_current.real - carry.dx),
                          lambda _: carry.xmax,
                          None)
    return carry

def update_outside_source(carry):
    
    def positive_run_fn(carry):
        carry.xmax = lax.cond(carry.dz2_last <= carry.rho2,
                              lambda _: carry.xmax.at[carry.yi].set(carry.z_current.real),
                              lambda _: carry.xmax,
                              None)
        carry.dx = -carry.incr
        carry.z_current = jnp.complex128(carry.x0 + 1j * carry.z_current.imag)
        carry.xmin = carry.xmin.at[carry.yi].set(carry.z_current.real + carry.dx)
        return carry
    
    def negative_run_fn(carry):
        def update_fn(carry):
            #carry.z_current += carry.dx
            return carry 

        def collect_fn(carry):
            carry.count_all += carry.count_x
            carry.area_x = carry.area_x.at[carry.yi].set(carry.count_x)
            carry.y = carry.y.at[carry.yi].set(carry.z_current.imag)
            carry.dys = carry.dys.at[carry.yi].set(carry.dy)
            return carry

        def break_fn(carry):
            carry.dys = carry.dys.at[carry.yi].set(-carry.dy) 
            carry.finish = jnp.bool_(True)
            return carry
        
        def check_counted_or_not_fn(carry):
            def counted_fn(carry):
                carry.area_x    = carry.area_x.at[carry.yi].set(0.0)
                carry.count_all = carry.count_all - carry.count_x
                carry.count_x   = 0.0 
                return carry
            
            y_index = int(carry.z_current.imag * carry.incr_inv + carry.max_iter)
            for j in jnp.arange(carry.Nindx[y_index]):
                ind = carry.indx[y_index][j]
                counted = (carry.xmin[carry.yi] < carry.xmax[ind]) & (carry.xmax[carry.yi] > carry.xmin[ind])
                carry = lax.cond(counted,
                                 counted_fn, 
                                 lambda carry: carry,
                                 carry)
            return carry
             
        def save_index_and_move_next_yrow(carry):
            y_index = int(carry.z_current.imag * carry.incr_inv + carry.max_iter)
            carry.indx  = carry.indx.at[y_index, carry.Nindx[y_index]].set(carry.yi)
            carry.Nindx = carry.Nindx.at[y_index].add(1.0)
            # prepare for the next row
            carry.yi += 1
            carry.dx = carry.incr
            carry.x0 = carry.xmax[carry.yi - 1]
            carry.z_current = jnp.complex128(carry.x0 + 1j * (carry.z_current.imag + carry.dy))
            carry.count_x = 0.0
            return carry

        carry.xmin = lax.cond(carry.dz2_last <= carry.rho2,
                              lambda _: carry.xmin.at[carry.yi].set(carry.z_current.real),
                              lambda _: carry.xmin,
                              None)
        
        x_larger_than_previous_xmin = (carry.z_current.real >= carry.xmin[carry.yi - 1] + carry.incr) 
        nothing_negative_run = (carry.yi != 0) & (carry.count_x == 0)
        cond_nothing_but_update = (x_larger_than_previous_xmin)&(nothing_negative_run)
        carry = lax.cond(cond_nothing_but_update,
                         update_fn,
                         collect_fn,
                         carry)
        
        carry = lax.cond((~cond_nothing_but_update)&(carry.count_x==0),
                         break_fn,
                         lambda carry: carry,
                         carry)

        carry = lax.cond((~cond_nothing_but_update)&(carry.count_x!=0),
                         check_counted_or_not_fn,
                         lambda carry: carry,
                         carry)
        
        carry = lax.cond((~cond_nothing_but_update)&(carry.count_x!=0),
                         save_index_and_move_next_yrow,
                         lambda carry: carry,
                         carry)
        return carry
    
    cond_positive_run    = carry.dx == carry.incr
    carry = lax.cond(cond_positive_run,
                     positive_run_fn,
                     negative_run_fn,
                     carry)
    return carry

"""
        def inner_cond(carry): 
            # nothing in negative run but should be extend  
            x_larger_than_previous_xmin = (carry.z_current.real >= carry.xmin[carry.yi - 1] + carry.incr) 
            nothing_negative_run = (carry.yi != 0) & (carry.count_x == 0)
            return nothing_negative_run & x_larger_than_previous_xmin & (~carry.finish) 
        
        def inner_true_fn(carry):
            carry.z_current += carry.dx
            return carry

        carry = lax.cond(inner_cond, 
                         inner_true_fn,
                         lambda carry: carry,
                         carry)
        
        carry.count_all += carry.count_x
        carry.area_x = carry.area_x.at[carry.yi].set(carry.count_x)
        carry.y = carry.y.at[carry.yi].set(carry.z_current.imag)
        carry.dys = carry.dys.at[carry.yi].set(carry.dy)

        carry.dys = lax.cond(carry.count_x == 0.0, # top in y, should be finished
                             lambda _: carry.dys.at[carry.yi].set(-carry.dy),
                             lambda _: carry.dys,
                             None)

        carry.finish = lax.cond(carry.count_x == 0.0, 
                                lambda _: jnp.bool_(True), 
                                lambda _: jnp.bool_(False),
                                None)
        
        def already_counted_fn(carry):
            carry.area_x    = carry.area_x.at[carry.yi].set(0.0)
            carry.count_all = carry.count_all - carry.count_x
            carry.count_x   = 0.0 
            return carry

        def not_counted_fn(carry):
            y_index = int(carry.z_current.imag * carry.incr_inv + carry.max_iter)
            for j in jnp.arange(carry.Nindx[y_index]):
                ind = carry.indx[y_index][j]
                counted = (carry.xmin[carry.yi] < carry.xmax[ind]) & (carry.xmax[carry.yi] > carry.xmin[ind])
                carry = lax.cond(counted,
                                 already_counted_fn, 
                                 lambda carry: carry,
                                 carry)
            carry.indx  = carry.indx.at[y_index, carry.Nindx[y_index]].set(carry.yi)
            carry.Nindx = carry.Nindx.at[y_index].add(1.0)
            # prepare for the next row
            carry.yi += 1
            carry.dx = carry.incr
            carry.x0 = carry.xmax[carry.yi - 1]
            carry.z_current = jnp.complex128(carry.x0 + 1j * (carry.z_current.imag + carry.dy))
            carry.count_x = 0.0
            return carry

        carry = lax.cond(inner_cond,
                         already_counted_fn,
                         not_counted_fn, 
                         carry)
        return carry

    return lax.cond(carry.dx == carry.incr, 
                    positive_run_fn, 
                    negative_run_fn,
                    carry) 
"""

@register_pytree_node_class
class CarryData:
    def __init__(self, yi: jnp.int32, indx: jnp.ndarray, Nindx: jnp.ndarray, xmin: jnp.ndarray, 
                 xmax: jnp.ndarray, area_x: jnp.ndarray, y: jnp.ndarray, dys: jnp.ndarray, 
                 z_current: jnp.complex128, x0: jnp.float, count_x: jnp.float, count_all: jnp.float, 
                 dz2: jnp.float, dz2_last: jnp.float, dx: jnp.float, finish: jnp.bool_, 
                 w_center: jnp.complex128, rho2: jnp.float, a: jnp.float, e1: jnp.float,
                 CM2MD: jnp.float, incr: jnp.float, incr_inv: jnp.float, max_iter: jnp.int32, 
                 nlenses: jnp.int32
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
        self.dz2_last = dz2_last
        self.dx = dx
        self.finish = finish
        # constants during functions
        self.w_center = w_center
        self.rho2 = rho2
        self.a = a
        self.e1 = e1
        self.CM2MD = CM2MD
        self.incr = incr
        self.incr_inv = incr_inv
        self.max_iter = max_iter
        self.nlenses = nlenses

    def tree_flatten(self):
        children = (self.yi, self.indx, self.Nindx, self.xmin, self.xmax, self.area_x, self.y, 
                    self.dys, self.z_current, self.x0, self.count_x, self.count_all, self.dz2, 
                    self.dz2_last, self.dx, self.finish, 
                    self.w_center, self.rho2, self.a, self.e1, self.CM2MD, self.incr, 
                    self.incr_inv, self.max_iter, self.nlenses)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)