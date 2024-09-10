import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.inverse_ray.image_area0 import image_area0

# θを0から2πの範囲に正規化する関数
def normalize_theta(theta):
    return jnp.mod(theta, 2 * jnp.pi)

@jax.jit
def merge_polar_intervals(polar_intervals, offset_r=1.0, offset_theta=0.1):
    # θを0から2πの範囲に正規化
    polar_intervals = polar_intervals.at[:, 1].set(normalize_theta(polar_intervals[:, 1]))
    polar_intervals = polar_intervals.at[:, 3].set(normalize_theta(polar_intervals[:, 3]))

    # 各極座標区間 (r_min, theta_min, r_max, theta_max) の範囲を拡張
    intervals = jnp.stack(
        [polar_intervals[:, 0] - offset_r, polar_intervals[:, 1] - offset_theta, 
         polar_intervals[:, 2] + offset_r, polar_intervals[:, 3] + offset_theta], axis=1
    )

    # r_min の昇順にソート
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]

    def merge_scan_fn(carry, next_interval):
        current_interval = carry

        # 半径方向の開始と終了
        r_start_max = jnp.maximum(current_interval[0], next_interval[0])
        r_start_min = jnp.minimum(current_interval[0], next_interval[0])
        r_end_max = jnp.maximum(current_interval[2], next_interval[2])
        r_end_min = jnp.minimum(current_interval[2], next_interval[2])

        # 角度方向の開始と終了 (θを0〜2πの範囲に正規化して比較)
        theta_start_max = jnp.maximum(current_interval[1], next_interval[1])
        theta_start_min = jnp.minimum(current_interval[1], next_interval[1])
        theta_end_max = jnp.maximum(current_interval[3], next_interval[3])
        theta_end_min = jnp.minimum(current_interval[3], next_interval[3])

        # 角度範囲が0〜2πをまたぐ場合の特別な処理
        overlap_exists = (r_start_max <= r_end_min) & (
            (theta_start_max <= theta_end_min) |
            (theta_start_min <= 0) & (theta_end_max >= 2 * jnp.pi)
        )

        # 重なりがあれば区間を統合
        updated_current_interval = jnp.where(
            overlap_exists,
            jnp.array([r_start_min, theta_start_min, r_end_max, theta_end_max]),
            next_interval
        )

        return updated_current_interval, updated_current_interval

    # 最初の区間で初期化し、後続の区間を順に処理
    _, merged_intervals = jax.lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])

    # 結果のマスク
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)

    return merged_intervals, mask

polar_intervals = jnp.array([
    [1.0, 0.1, 2.0, 1.0],  # 半径1-2, 角度0.1-1.0
    [1.5, 0.8, 3.0, 1.2],  # 半径1.5-3, 角度0.8-1.2 (一部重なる)
    [2.5, 6.0, 4.0, 0.2],  # 半径2.5-4, 角度6.0-0.2 (2πまたぎ)
    [3.5, 5.8, 5.0, 6.1],  # 半径3.5-5, 角度5.8-6.1 (6.0-0.2と重なる)
])

# テストする極座標区間マージ関数を実行
merged_intervals, mask = merge_polar_intervals(polar_intervals)

# 結果の表示
print("Merged intervals:")
print(merged_intervals)

print("Mask:")
print(mask)
