import jax.numpy as jnp
from jax import tree_util
from jax.tree_util import register_pytree_node_class
import jax
from jax import lax, jit
from functools import partial
from ..point_source import lens_eq, _images_point_source

def source_profile_limb1(dz2, u1=0.5):
    mu = jnp.sqrt(1.0 - dz2)
    return 1 - u1 * (1.0 - mu)

@partial(jit, static_argnames=("nlenses"))
def image_area0(w_center, rho, z_init, dy, carry, nlenses=2, **_params):
    """
    Calculate the image area for a given initial complex coordinate and step size in the y-direction.

    Args:
        w_center (complex): Complex coordinate of the source center.
        rho (float): Radius of the source.
        z_init (complex): Initial complex coordinate for the lens equation.
        dy (float): Step vector toward the y-direction.
        carry (tuple): Carry data containing intermediate results.
        nlenses (int, optional): Number of lenses. Default is 2.
        **_params: Additional parameters for the lens model, including mass ratio (q) and separation (s).

    Returns:
        tuple: Calculated area for the given parameters and updated carry data.
    """
    yi, indx, Nindx, xmin, xmax, area_x, y, dys = carry 
    max_iter = len(dys) // 2
    q, s = _params["q"], _params["s"]
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
    finish = jnp.bool_(True)

    carry_init = CarryData(yi=yi, indx=indx, Nindx=Nindx, xmin=xmin, 
                            xmax=xmax, area_x=area_x, y=y, dys=dys,
                            z_current=z_current, x0=x0, count_x=count_x, 
                            count_all=count_all, dz2=dz2, dz2_last=dz2, dx=dx, finish=finish,
                            w_center=w_center, rho2=rho2, a=a, e1=e1, CM2MD=CM2MD, 
                            incr=incr, incr_inv=incr_inv, max_iter=max_iter, nlenses=nlenses, 
                            dy=dy)   
    
    result = lax.while_loop(cond_fun = lambda carry: carry.finish, 
                            body_fun = update_dy, 
                            init_val = carry_init)
    
    #jax.debug.print('{}', result.dy)
    carry_return = (result.yi, result.indx, result.Nindx, result.xmin, 
                    result.xmax, result.area_x, result.y, result.dys)

    return result.count_all, carry_return  

#@partial(jit, static_argnames=("nlenses"))
def update_dy(carry):
    #jax.debug.print('{} {}', carry.yi, carry.finish)
    z_current_mid = carry.z_current + carry.CM2MD
    zis_mid = lens_eq(z_current_mid, a=carry.a, e1=carry.e1)
    #zis_mid = lens_eq(z_current_mid, a=carry.a, e1=carry.e1, nlenses=carry.nlenses)
    zis = zis_mid - carry.CM2MD
    carry.dz2_last = carry.dz2
    dz = jnp.abs(carry.w_center - zis)
    carry.dz2 = dz**2

    carry = lax.cond(carry.dz2 <= carry.rho2,
                     update_inside_source,
                     update_outside_source,
                     carry)

    carry.z_current = lax.cond(carry.finish,
                               lambda carry: carry.z_current + carry.dx,
                               lambda carry: carry.z_current,
                               carry)
    return carry

def update_inside_source(carry):
    #jax.debug.print('update_inside_source yi={} dx = {} y={} z.real={} count_x={}', 
    #                carry.yi, carry.dx, carry.y[carry.yi], carry.z_current.real, carry.count_x)
    # first step in negative run
    carry.xmax = lax.cond((carry.dx == -carry.incr) & (carry.count_x == 0.0),
                          lambda _: carry.xmax.at[carry.yi].set(carry.z_current.real + carry.incr),
                          #lambda _: carry.xmax.at[carry.yi].set(carry.z_current.real - carry.dx),
                          lambda _: carry.xmax,
                          None)
    count_eff = source_profile_limb1(carry.dz2)
    carry.count_x += count_eff #.astype(float)
    return carry

def update_outside_source(carry):
    #jax.debug.print('update_outside_source yi={} dx = {} y={} z.real={} count_x={}', 
    #                carry.yi, carry.dx, carry.y[carry.yi], carry.z_current.real, carry.count_x)
    def positive_run_fn(carry):
        carry.xmax = lax.cond(carry.dz2_last <= carry.rho2,
                              lambda _: carry.xmax.at[carry.yi].set(carry.z_current.real),
                              lambda _: carry.xmax,
                              None)
        # switch to negative run
        carry.dx = -carry.incr
        carry.z_current = jnp.complex128(carry.x0 + 1j * carry.z_current.imag)
        carry.xmin = carry.xmin.at[carry.yi].set(carry.z_current.real + carry.dx)
        return carry
    
    def negative_run_fn(carry):
        def collect_fn(carry):
            carry.count_all += carry.count_x
            carry.area_x = carry.area_x.at[carry.yi].set(carry.count_x)
            carry.y      = carry.y.at[carry.yi].set(carry.z_current.imag)
            carry.dys    = carry.dys.at[carry.yi].set(carry.dy) 
            return carry

        def break_fn(carry):
            carry.dys = carry.dys.at[carry.yi].set(-carry.dy) 
            carry.finish = jnp.bool_(False)
            return carry
        
        def check_overlap_fn(carry):
            #jax.debug.print("check_counted_or_not_fn")
            def overlap_fn(carry):
                #y_index = jnp.int32(carry.z_current.imag * carry.incr_inv + carry.max_iter)
                #jax.debug.print('overlap_fn yi={} y={} dys={} y_index={}', carry.yi, carry.y[carry.yi], carry.dys[carry.yi], y_index)
                #jax.debug.print('           xmin={} xmax={}', carry.xmin[carry.yi], carry.xmax[carry.yi])
                #jax.debug.print('           xmin_rows={}'   , xmins_same_row)
                #jax.debug.print('           xmax_rows={}'   , xmaxs_same_row)
                #jax.debug.print('           indices  ={}'   , indices)
                carry.area_x = carry.area_x.at[carry.yi].set(0.0)
                carry.count_all = carry.count_all - carry.count_x
                carry.count_x   = 0.0 
                carry.finish = jnp.bool_(False)
                return carry

            y_index = jnp.int32(carry.z_current.imag * carry.incr_inv + carry.max_iter)
            indices = carry.indx[y_index]
            xmaxs_same_row = jnp.where(indices==0, -jnp.inf, carry.xmax[indices])
            xmins_same_row = jnp.where(indices==0, jnp.inf,  carry.xmin[indices])
            
            """
            2024/08/07 This is critical
            if the image is very elongated in x-direction, overlap_mask may not work.
            """
            tuned_fac = 2.0
            #jax.debug.print('indices={} xmax={} xmax={}', indices, xmax_row, xmin_row) 
            overlap_mask = \
                (carry.xmin[carry.yi] - tuned_fac * carry.incr < xmaxs_same_row) \
                & (carry.xmax[carry.yi] + tuned_fac * carry.incr > xmins_same_row)\
                & jnp.all(indices != carry.yi - 1) # because the previous row can be wrongly added. I'm not sure why.
            carry = lax.cond(jnp.any(overlap_mask),
                             overlap_fn, 
                             lambda carry: carry, 
                             carry)
            return carry
             
        def save_index_and_move_next_yrow(carry):
            #jax.debug.print('save_index_and_move_next_yrow')
            y_index = jnp.int_(carry.z_current.imag * carry.incr_inv + carry.max_iter)
            carry.indx = carry.indx.at[y_index, carry.Nindx[y_index]].set(carry.yi)
            carry.Nindx = carry.Nindx.at[y_index].add(1.0)
            #jax.debug.print('yi={} y={} y_index={} dys={} xmin={} xmax={}', 
            #                carry.yi, carry.y[carry.yi], y_index, carry.dys[carry.yi], carry.xmin[carry.yi], carry.xmax[carry.yi]) 
            # prepare for the next row
            carry.yi += 1
            carry.dx  = carry.incr
            carry.x0  = carry.xmax[carry.yi - 1]
            carry.z_current = jnp.complex128(carry.x0 - carry.dx + 1j * (carry.z_current.imag + carry.dy))
            carry.count_x = 0.0
            #jax.debug.print('y={} xmin={} xmax{}', 
            #                carry.y[carry.yi-1], carry.xmin[carry.yi-1], carry.xmax[carry.yi-1] ) 
            return carry

        # whether the previous z is inside or not
        carry.xmin = lax.cond(carry.dz2_last <= carry.rho2,
                              lambda _: carry.xmin.at[carry.yi].set(carry.z_current.real),
                              lambda _: carry.xmin,
                              None)

        # condition in negative run
        previous_yi_is_connected = jnp.abs(carry.z_current.imag - carry.y[carry.yi -1]) < 2.0 * carry.incr
        x_larger_than_previous_xmin = (carry.z_current.real >= carry.xmin[carry.yi - 1] + carry.incr) & (previous_yi_is_connected)
        nothing_negative_run = (carry.yi != 0) & (carry.count_x == 0)
        cond_nothing_but_update = (x_larger_than_previous_xmin)&(nothing_negative_run)

        carry = lax.cond((~cond_nothing_but_update)&(carry.finish),
                         collect_fn,
                         lambda carry: carry, # continue
                         carry)

        # break if count_x==0 and xmin is small enough. 
        carry = lax.cond((~cond_nothing_but_update)&(carry.count_x==0)&(carry.finish),
                         break_fn,
                         lambda carry: carry,
                         carry)

        # check if the count is overlapped with previous ones.
        carry = lax.cond((~cond_nothing_but_update)&(carry.finish),
                         check_overlap_fn,
                         lambda carry: carry,
                         carry)

        # memorize this row and move to the next row
        carry = lax.cond((~cond_nothing_but_update)&(carry.finish),
                         save_index_and_move_next_yrow,
                         lambda carry: carry,
                         carry)
        return carry
    
    carry = lax.cond(carry.dx == carry.incr,
                     positive_run_fn,
                     negative_run_fn,
                     carry)
    return carry

@register_pytree_node_class
class CarryData:
    def __init__(self, yi: jnp.int_, indx: jnp.ndarray, Nindx: jnp.ndarray, xmin: jnp.ndarray, 
                 xmax: jnp.ndarray, area_x: jnp.ndarray, y: jnp.ndarray, dys: jnp.ndarray, 
                 z_current: jnp.complex128, x0: jnp.float_, count_x: jnp.float_, count_all: jnp.float_, 
                 dz2: jnp.float_, dz2_last: jnp.float_, dx: jnp.float_, finish: jnp.bool_, 
                 w_center: jnp.complex128, rho2: jnp.float_, a: jnp.float_, e1: jnp.float_,
                 CM2MD: jnp.float_, incr: jnp.float_, incr_inv: jnp.float_, max_iter: jnp.int16, 
                 nlenses: jnp.int16, dy: jnp.float_
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
        self.dy = dy

    def tree_flatten(self):
        children = (self.yi, self.indx, self.Nindx, self.xmin, self.xmax, self.area_x, self.y, 
                    self.dys, self.z_current, self.x0, self.count_x, self.count_all, self.dz2, 
                    self.dz2_last, self.dx, self.finish, 
                    self.w_center, self.rho2, self.a, self.e1, self.CM2MD, self.incr, 
                    self.incr_inv, self.max_iter, self.nlenses, self.dy)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)