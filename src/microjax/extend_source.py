import jax.numpy as jnp
from microjax.poly_solver import poly_roots_EA_multi as poly_roots
from microjax.point_source import _lens_eq_binary

def source_profile_limb1(dz, u1=0.0):
    mu = jnp.sqrt(1.0 - dz*dz)
    return 1 - u1 * (1.0 - mu)

# very slow and cannot differentiable
def image_area0_binary(w_center, z_init, q, s, rho, dy, carry): 
    """ 
    Auxiliary function to calculate area of an image for binary lens system by inverse-ray shooting.

    Args:
        w_center (complex): Position of the source in the complex plane, center-of mass coordinate.
        z_init (complex): Initial position of the image point, center-of mass coordinate. 
        q (float): Mass ratio of the binary lens.
        s (float): Separation between the two lenses.
        rho (float): Radius of the source.
        dy (float): Step size in the y-direction.
        carry (tuple): Tuple containing arrays and indices for ongoing calculations.

    Returns:
        float: Total brightness of the image area, adjusted for limb darkening.
        tuple: Updated carry containing intermediate results for continued calculations.
    """
    yi, indx, Nindx, xmax, xmin, area_x, y, dys = carry 
    max_iter = int(len(dys) / 2.0)
    CM2MD = - 0.5*s*(1 - q)/(1 + q) 
    z_current = z_init
    x0   = z_init.real
    a    = 0.5 * s
    e1   = q / (1.0 + q) 
    dz2  = 9999.9999
    incr      = jnp.abs(dy)
    incr_inv  = 1.0 / incr
    dx        = incr 
    count_x   = 0.0
    count_all = 0.0
    rho2      = rho * rho
    
    while True:
        z_current_mid = z_current + CM2MD
        zis_mid = _lens_eq_binary(z_current_mid, a=a, e1=e1) # inversed point from image into source, mid-point coordinate
        zis = zis_mid - CM2MD
        dz2_last = dz2
        dz  = jnp.abs(w_center - zis)
        dz2 = dz**2
        if dz2 <= rho2: # inside of the source
            if dx == -incr and count_x == 0.0: # update xmax value if negative run
                xmax = xmax.at[yi].set(z_current.real - dx)
            count_eff = source_profile_limb1(dz) # brightness with limb-darkening
            count_x   += float(count_eff)
        else: 
            if dx == incr: # positive run outside of the source
                if dz2_last <= rho2: 
                    xmax = xmax.at[yi].set(z_current.real) # store the previous ray as xmax
                # update to negative
                dx = -incr 
                z_current = jnp.complex128(x0 + 1.0j * z_current.imag)
                xmin = xmin.at[yi].set(z_current.real + dx) # set xmin in positive run
            else: # negative run with outside of the source 
                if dz2_last <= rho2: # if previous ray is inside
                    xmin = xmin.at[yi].set(z_current.real) # set xmin in negative run 
                if z_current.real >= xmin[yi-1] - dx and yi!=0 and count_x==0: # nothing in negative run
                    z_current = z_current + dx
                    continue
                # collect numbers in the current y coordinate
                count_all += count_x
                area_x = area_x.at[yi].set(count_x)
                y      = y.at[yi].set(z_current.imag)
                dys    = dys.at[yi].set(dy)
                if count_x == 0.0: # This means top in y
                    dys = dys.at[yi].set(-dy)
                    #print("top in y!")
                    break
                # check if this y is already counted
                #y_index = int(z_current.imag * incr_inv) #the index based on the current y coordinate
                y_index = int(z_current.imag * incr_inv + max_iter) #the index based on the current y coordinate
                for j in range(Nindx[y_index]):
                    ind = indx[y_index][j]
                    #ind = indx[y_index][j+1]
                    """
                    if xmin[yi] + incr < xmax[ind] and xmax[yi] - incr > xmin[ind]: # already counted.
                        carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
                        print("already counted...")
                        return count_all - count_x, carry
                    """
                # save index yi if counted
                indx = indx.at[y_index, Nindx[y_index]].set(yi)
                Nindx = Nindx.at[y_index].add(1)
                print("new yi: yi=%d dx=%.3f xmin=%.3f xmax=%.3f y=%.3f dys=%.3f count_x=%d count_all=%d z.i=%.3f"
                      %(yi, dx, xmin[yi], xmax[yi], y[yi], dys[yi], count_x, count_all, z_current.imag))
                # move next y-row 
                yi       += 1
                dx        = incr               # switch to positive run
                x0        = xmax[yi-1]         # starting x in next negative run.  
                z_current = jnp.complex128(x0 + dx + 1j * z_current.imag + 1.0j * dy)  # starting point in next positive run.
                count_x = 0.0
        # update the z value 
        z_current = z_current + dx
    
    carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
    print("last yi: yi=%d dx=%.3f xmin=%.3f xmax=%.3f y=%.3f dys=%.3f count_x=%d count_all=%d z.i=%.3f"
          %(yi, dx, xmin[yi-1], xmax[yi-1], y[yi-1], dys[yi-1], count_x, count_all, z_current.imag))
    return count_all, carry


def image_area_binary(w_center, z_inits, q, s, rho, NBIN=10, max_iter=100000):
    """ 
    Compute the area of the image formed by a binary lens system.

    Args:
        w_center (complex): Position of the source in the complex plane.
        z_inits (array_like): Initial guesses for the positions of the images in the complex plane.
        q (float): Mass ratio of the binary lens components.
        s (float): Separation between the two lenses.
        rho (float): Radius of the source.
        NBIN (int, optional): 
            Number of bins defined by rho/BIN, resolution of inverse-ray shooting in rho.
        max_iter (int, optional): 
            Maximum number of iterations for the algorithm.
            (Resolution in RE) = 1 / incr = NBIN / rho

    Returns:
        float: Total area of all images produced by the lens system.
    """
    # Initialize arrays to store intermediate results and indices
    indx   = jnp.zeros((max_iter * 2, 4), dtype=int)
    Nindx  = jnp.zeros((max_iter * 2,),   dtype=int)
    xmax   = jnp.zeros((max_iter * 4,))
    xmin   = jnp.zeros((max_iter * 4,))
    area_x = jnp.zeros((max_iter * 4,))
    y      = jnp.zeros((max_iter * 4,))
    dys    = jnp.zeros((max_iter * 4,))
    area   = 0.0

    nimage  = len(z_inits)
    overlap = jnp.zeros(6) # binary-lens
    incr    = rho / NBIN  
    incr2   = 0.5 * incr 
    incr2margin = incr2 * 1.01  
    
    if rho <= 0:
        return 0.0
    
    area_image = jnp.zeros(nimage)
    for i in range(nimage):
    #for i in range(1, nimage+1):
        if overlap[i] == 1:
            continue
        area_i = 0.0
        # search image toward +y
        yi = 0
        dy = incr
        z_init = z_inits[i]
        xmin = xmin.at[0].set(z_init[i].real)
        xmax = xmax.at[0].set(z_init[i].real)
        carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
        area_y_plus, carry = image_area0_binary(w_center, z_init, q, s, dy, carry)
        area_i += area_y_plus

        # search image toward -y
        (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry  
        dy       = -incr
        z_init   = jnp.complex128(xmax[0], z_inits[i].imag + dy)
        xmin = xmin.at[yi].set(xmin[0])
        xmax = xmax.at[yi].set(xmax[0])
        y    = y.at[yi].set(y[0])
        dys  = dys.at[yi].set(dy) # remember new direction
        yi  += 1
        carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
        area_y_minus, carry = image_area0_binary(w_center, z_init, q, s, dy, carry)
        area_i += area_y_minus

        # search extra image beyond the boundary
        (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry  
        nyi = yi
        area_bound = 0.0
        for j in range(nyi):
            dxmax = xmax[j + 1] - xmax[j]
            dxmin = xmin[j + 1] - xmin[j]
            if area_x[j+1] > 0.0:
                if dxmax > 1.1 * incr:
                    z_init = jnp.complex128(xmax[j + 1] + 1j *  y[j])
                    xmin   = xmin.set[yi].set(xmax[j])
                    xmax   = xmax.set[yi].set(xmax[j + 1])
                    dy     = -dys[j]
                    dys    = dys.at[yi].set(dy)
                    yi    += 1
                    
                    carry        = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
                    area0, carry = image_area0_binary(w_center, z_init, q, s, dy, carry)
                    area_i      += area0
                    area_bound  += area0
                    if area0 <= 0:
                        yi -= -1
                if dxmin > 1.1 * incr:
                    z_init = jnp.complex128(xmax[j + 1] - incr + 1j *  y[j])
                    xmin   = xmin.set[yi].set(xmin[j])
                    xmax   = xmax.set[yi].set(xmin[j + 1])
                    dy     = dys[j]
                    dys    = dys.at[yi].set(dy)
                    yi    += 1

                    carry        = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
                    area0, carry = image_area0_binary(w_center, z_init, q, s, dy, carry)
                    area_i      += area0
                    area_bound  += area0
                    if area0 <= 0:
                        yi -= -1
                if dxmin < -1.1 * incr:
                    z_init = jnp.complex128(xmin[j] - incr + 1j *  y[j])
                    xmin   = xmin.set[yi].set(xmin[j + 1])
                    xmax   = xmax.set[yi].set(xmin[j])
                    dy     = -dys[j]
                    dys    = dys.at[yi].set(dy)
                    yi    += 1

                    carry        = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
                    area0, carry = image_area0_binary(w_center, z_init, q, s, dy, carry)
                    area_i      += area0
                    area_bound  += area0
                    if area0 <= 0:
                        yi -= -1
            # search boundary again if new image area was found
            if j == nyi -1 and area_bound > 0.0 and yi > nyi:
                nyi = yi
        
        area += area_i # total area of all images
        area_image = area_image.at[i].set(area_i) # area of image i

        # search overlapping images
        for ii in range(nimage): #  roop for images
        #for ii in range(1, nimage + 1): #  roop for images
            if ii == i:
                continue
            for j in range(nyi): # roop for boundaries
                if area_x[j] <= 0.0:
                    continue
                if z_inits[ii].imag >= y[j] - incr2margin and z_inits[ii].imag <= y[j] + incr2margin:
                    if z_inits[ii].real >= xmin[j] - incr2margin and z_inits[ii].real <= xmax[j] + incr2margin:
                        if ii < i: # already searched image
                            area -= area_image[ii]
                        else:
                            overlap = overlap.at[ii].set(1.0)
                        break
        # initialize index for checking roop image
        for j in range(nyi):
            index = int(y[j] / incr + max_iter)
            Nindx = Nindx.at(index).set(0)
    
    return area
