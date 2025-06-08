# wheel 同梱ライブラリ以外を一切見せない
export LD_LIBRARY_PATH=
python - <<'PY'
import jax, jax.numpy as jnp
x = jnp.eye(4096, dtype=jnp.float32)
print((x @ x).block_until_ready().shape)
PY

