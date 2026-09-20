"""Is the GPU usable, and what batch size does EfficientNet-B3 at 512 train at?"""
import sys
import time
import tensorflow as tf
from tensorflow import keras

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
if len(sys.argv) > 2 and sys.argv[2] == "mixed":
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("policy: mixed_float16", flush=True)
m = keras.applications.EfficientNetB3(include_top=False, weights=None, input_shape=(size, size, 3), pooling="avg")
head = keras.layers.Dense(1, dtype="float32")
opt = keras.optimizers.AdamW(1e-4)
variables = m.trainable_variables + head.trainable_variables


@tf.function(jit_compile=("xla" in sys.argv))
def step(x, y):
    with tf.GradientTape() as tape:
        pred = head(m(x, training=True))
        loss = tf.reduce_mean(tf.square(pred - y))
    grads = tape.gradient(loss, variables)
    opt.apply_gradients(zip(grads, variables))
    return loss


for batch in (4, 8, 12, 16):
    try:
        x = tf.random.normal((batch, size, size, 3)); y = tf.zeros((batch, 1))
        step(x, y)  # trace + JIT
        t = time.perf_counter()
        for _ in range(5):
            step(x, y)
        dt = (time.perf_counter() - t) / 5
        print(f"batch {batch}: {dt * 1000:.0f} ms/step, {batch / dt:.1f} img/s", flush=True)
    except tf.errors.ResourceExhaustedError:
        print(f"batch {batch}: OOM", flush=True)
        break
