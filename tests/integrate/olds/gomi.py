""" 
import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
from microjax.inverse_ray.merge_area import calc_source_limb, calculate_overlap_and_range 
import jax.numpy as jnp
from jax import jit, vmap

import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnums=(2, 3, 4, 5, 6, ))
def mag_binary(w_center, rho, th_resolution=100, Nlimb=100, offset_r=1.0, offset_th=5.0, GRID_RATIO=5, **_params):
    q, s = _params["q"], _params["s"]
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q)
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)
    r_use = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    # 5 images for binary-lens
    r_use = r_use[jnp.argsort(r_use[:, 1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:, 1])][-5:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2 * jnp.pi)

    resolution = resolution * GRID_RATIO
    r_grid_normalized = jnp.linspace(0, 1, resolution + 1)
    th_grid_normalized = jnp.linspace(0, 1, th_resolution + 1)
    r_mesh_norm, th_mesh_norm = jnp.meshgrid(r_grid_normalized, th_grid_normalized, indexing='ij')

    @jit
    def compute_for_range(r_range, th_range):
        in_mask = jnp.any((r_limb >= r_range[0]) & (r_limb <= r_range[1]) &
                          (th_limb >= th_range[0]) & (th_limb <= th_range[1]))

        def compute_if_in():
            dr = (r_range[1] - r_range[0]) / resolution
            dth = (th_range[1] - th_range[0]) / th_resolution
            r_mesh = r_mesh_norm * (r_range[1] - r_range[0]) + r_range[0]
            th_mesh = th_mesh_norm * (th_range[1] - th_range[0]) + th_range[0]
            z_mesh = r_mesh * (jnp.cos(th_mesh) + 1j * jnp.sin(th_mesh))
            z_mesh_flat = z_mesh.reshape(-1)
            image_mesh = lens_eq(z_mesh_flat - shifted, **_params)
            distances = jnp.abs(image_mesh - w_center_shifted).reshape(z_mesh.shape)
            # distance from source center in each vertex of each cell
            r00 = r_mesh[:-1, :-1]
            r11 = r_mesh[1:, 1:]
            x00 = r_mesh[:-1, :-1] * jnp.cos(th_mesh[:-1, :-1])
            y00 = r_mesh[:-1, :-1] * jnp.sin(th_mesh[:-1, :-1])
            x10 = r_mesh[1:, :-1] * jnp.cos(th_mesh[1:, :-1])
            y10 = r_mesh[1:, :-1] * jnp.sin(th_mesh[1:, :-1])
            x01 = r_mesh[:-1, 1:] * jnp.cos(th_mesh[:-1, 1:])
            y01 = r_mesh[:-1, 1:] * jnp.sin(th_mesh[:-1, 1:])
            x11 = r_mesh[1:, 1:] * jnp.cos(th_mesh[1:, 1:])
            y11 = r_mesh[1:, 1:] * jnp.sin(th_mesh[1:, 1:])
            
            d00 = distances[:-1, :-1] 
            d10 = distances[1:, :-1]
            d01 = distances[:-1, 1:]
            d11 = distances[1:, 1:] 
            v00 = d00 - rho
            v10 = d10 - rho 
            v01 = d01 - rho
            v11 = d11 - rho
            # v?? <= 0 means that is inside of source.
            s00 = (v00 <= 0).astype(int)
            s10 = (v10 <= 0).astype(int)
            s01 = (v01 <= 0).astype(int)
            s11 = (v11 <= 0).astype(int)
            cases = s00 + s10 * 2 + s11 * 4 + s01 * 8
            
            cell_area = 0.5 * (r00 + r11) * dr * dth

            eps = 1e-10
            def find_crossing(y1, y2):
                "find x at y=0 given (0.0, y1) and (1.0, y2)"
                return jnp.clip(y1 / (y1 - y2 + eps), 0.0, 1.0)

            def area_fraction(cases, v00, v10, v11, v01):
                """
                approximate partial spaces in each single element in polar coordinate.
                Using extended Marching Squares method.
                We do know that the function behaves as sqrt{x - xL} where xL is the boundary. 
                x01 --- x11
                 |       |
                 |       |     
                x00 --- x10   -> dr           
                """
                # crossing points
                x0010 = find_crossing(v00, v10)
                x0001 = find_crossing(v00, v01)
                x0111 = find_crossing(v01, v11)
                x1011 = find_crossing(v10, v11)

                area = jnp.zeros_like(cell_area)
                area = jnp.where(cases == 0, 0.0, area)
                area = jnp.where(cases == 1, 0.5 * cell_area * x0010 * x0001, area)
                area = jnp.where(cases == 2, 0.5 * cell_area * (1 - x0010) * x1011, area)
                area = jnp.where(cases == 3, cell_area * 0.5 * (1 - x0010), area)
                area = jnp.where(cases == 4, 0.5 * cell_area * (1 - x1011) * (1 - x0111), area)
                area = jnp.where(cases == 5, 0.5 * cell_area, area)
                area = jnp.where(cases == 6, 0.5 * cell_area * (1 - x0010), area)
                area = jnp.where(cases == 7, cell_area * 0.5, area)
                area = jnp.where(cases == 8, 0.5 * cell_area * x0001 * (1 - x0111), area)
                area = jnp.where(cases == 9, 0.5 * cell_area * x0001, area)
                area = jnp.where(cases == 10, 0.5 * cell_area, area)
                area = jnp.where(cases == 11, 0.75 * cell_area, area)
                area = jnp.where(cases == 12, 0.5 * cell_area * (1 - x0111), area)
                area = jnp.where(cases == 13, 0.75 * cell_area, area)
                area = jnp.where(cases == 14, cell_area * 0.75, area)
                
                #case 1: 00 が内部で、他は外部にある場合
                #area = jnp.where(cases == 1, x0010 * x0001 * 0.5 * dr * dth, area)
                #case 2: 10 が内部で、他は外部にある場合
                #case 3: 00 と 10 が内部で、11 と 01 が外部にある場合
                #case 4: 11 が内部で、他は外部にある場合
                #case 5: 00 と 11 が内部で、10 と 01 が外部にある場合
                #case 6: 10 と 11 が内部で、00 と 01 が外部にある場合
                #case 7: 00, 10, 11 が内部で、01 が外部にある場合
                #case 8: 01 が内部で、他は外部にある場合
                #case 9: 00 と 01 が内部で、10 と 11 が外部にある場合
                #case 10: 10 と 01 が内部で、00 と 11 が外部にある場合
                #case 11: 00, 10, 01 が内部で、11 が外部にある場合
                #case 12: 11 と 01 が内部で、00 と 10 が外部にある場合
                #case 13: 00, 11, 01 が内部で、10 が外部にある場合
                #case 14: 10, 11, 01 が内部で、00 が外部にある場合
                area = jnp.where(cases == 15, cell_area, area)
                return area
            area = area_fraction(cases, v00, v10, v11, v01)
            total_area = jnp.sum(area)

            return total_area

        return jnp.where(in_mask, compute_if_in(), 0.0)

    compute_vmap = jit(vmap(vmap(compute_for_range, in_axes=(None, 0)), in_axes=(0, None)))
    image_areas = compute_vmap(r_use, th_use)
    total_area = jnp.sum(image_areas)

    magnification = total_area / (rho**2 * jnp.pi)
    return magnification

    def compute_for_range(r_range, th_range):
        in_mask = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) &
                          (th_limb > th_range[0]) & (th_limb < th_range[1]))
        def compute_if_in():
            dr = (r_range[1] - r_range[0]) / resolution
            dth = (th_range[1] - th_range[0]) / (resolution * GRID_RATIO)
            r_mesh = r_mesh_norm * (r_range[1] - r_range[0]) + r_range[0]
            th_mesh = th_mesh_norm * (th_range[1] - th_range[0]) + th_range[0]
            z_mesh = r_mesh * (jnp.cos(th_mesh) + 1j * jnp.sin(th_mesh))
            image_mesh = lens_eq(z_mesh - shifted, **_params)
            distances  = jnp.abs(image_mesh - w_center_shifted)
            #image_mask = distances < rho
            def integrate_1d(r_line, th_line, d_line):
                in_source = (d_line - rho < 0.0).astype(float)
                r0 = r_line[0]
                th0, th1 = th_line[:-1], th_line[1:]
                in0 = in_source[:-1]
                in1 = in_source[1:]
                segment_inside = (in0 == 1) & (in1 == 1)
                segment_crossing = (in0 != in1)

                def compute_crossing_area(th0):
                    th_fine  = th0 + dth * th_fine_grid
                    dth_fine = dth / num_fine 
                    z_fine = r0 * (jnp.cos(th_fine) + 1j * jnp.sin(th_fine))
                    image_mesh_fine = lens_eq(z_fine - shifted, **_params)
                    distances_fine = jnp.abs(image_mesh_fine - w_center_shifted)
                    in_source_fine = (distances_fine - rho < 0.0).astype(float)
                    in0_fine, in1_fine = in_source_fine[:-1], in_source_fine[1:]
                    d0_fine, d1_fine = distances_fine[:-1], distances_fine[1:] 

                    frac = jnp.clip((rho - d0_fine) / (d1_fine - d0_fine), 0.0, 1.0)
                    area_inside_fine = jnp.where((in0_fine == 1) & (in1_fine == 1), 1.0, 0.0)
                    area_crossing_in2out_fine = jnp.where((in0_fine == 1) & (in1_fine == 0), frac, 0.0)
                    area_crossing_out2in_fine = jnp.where((in0_fine == 0) & (in1_fine == 1), 1.0 - frac, 0.0)

                    area = jnp.sum(area_inside_fine + area_crossing_in2out_fine + area_crossing_out2in_fine)
                    return area * r0 * dth_fine
                
                # modified 
                #area_crossing = vmap(lambda th0, crossing_flag: lax.cond(
                #    crossing_flag, 
                #    lambda _:compute_crossing_area(th0), 
                #    lambda _: 0.0,
                #    operand=None))(th0, segment_crossing)

                area_inside = segment_inside * r0 * dth # mid-point approximation, 0.5*(f0+f1)
                frac = jnp.clip((rho - d_line[:-1]) / (d_line[1:] - d_line[:-1]), 0.0, 1.0)
                area_crossing = segment_crossing * (in0 * frac + in1 * (1.0 - frac)) * r0 * dth
                
                total_area = jnp.sum(area_inside) + jnp.sum(area_crossing)
                return total_area
                
            int_1d = vmap(integrate_1d, in_axes=(0, 0, 0,))
            areas = int_1d(r_mesh, th_mesh, distances)
            area = dr * jnp.sum(areas)
            return area
        
        return jnp.where(in_mask, compute_if_in(), 0.0)
"""