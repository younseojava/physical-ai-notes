import argparse
import time
import quadrants as qd

qd.init(arch=qd.amdgpu)

parser = argparse.ArgumentParser()
parser.add_argument("--use-field", action="store_true")
args = parser.parse_args()

use_field = args.use_field
if use_field:
    V = qd.field
    ParamType = qd.Template
else:
    V = qd.ndarray
    ParamType = qd.types.NDArray[qd.i32, 1]

@qd.kernel
def copy_memory(N: int, a: ParamType, b: ParamType) -> None:
    for n in range(N):
        b[n % 100] = a[n % 100]

N = 20_000
a = V(qd.i32, (100,))
b = V(qd.i32, (100,))

# warmup
copy_memory(N, a, b)

num_its = 1000

qd.sync()
start = time.time()
for it in range(num_its):
    copy_memory(N, a, b)
qd.sync()
end = time.time()
iteration_time = (end - start) / num_its * 1_000_000
print("iteration time", iteration_time, "us")

"""
For NVidia example, 5090 gpu: ndarray vs. field has 28% improvement:  29.5 us vs. 21.4 us.

For MI325X (amdgpu), not much of difference between ndarray and field: 40.34 us vs 40.32 us.
  -->  need to optimize more to see the difference, as well as improve ndarray time also.
"""