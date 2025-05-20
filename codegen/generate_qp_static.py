import numpy as np
import os

npy_dir = "output/"
header_file = "qp_data_static.h"

arrays = {
    "H_p": "H_csc_p.npy",
    "H_i": "H_csc_i.npy",
    "H_x": "H_csc_x.npy",
    "q": "q.npy",
    "A_p": "A_csc_p.npy",
    "A_i": "A_csc_i.npy",
    "A_x": "A_csc_x.npy",
    "b": "b.npy",
    "C_p": "C_csc_p.npy",
    "C_i": "C_csc_i.npy",
    "C_x": "C_csc_x.npy",
    "l": "l.npy",
    "u": "u.npy"
}

def to_cpp_array(name, arr):
    cpp_type = "int" if arr.dtype in [np.int32, np.int64] else "double"
    values = ", ".join(map(str, arr.tolist()))
    return f"constexpr {cpp_type} {name}[{len(arr)}] = {{ {values} }};\n"

with open(header_file, "w") as f:
    f.write("// Auto-generated QP data for static NASOQ use\n")
    f.write("#pragma once\n\n")
    f.write("#include <cstddef>\n\n")
    f.write("namespace qp_data {\n\n")

    for var, fname in arrays.items():
        path = os.path.join(npy_dir, fname)
        if not os.path.exists(path):
            print(f"Warning: file not found: {path}")
            continue
        arr = np.load(path)
        f.write(to_cpp_array(var, arr))
        f.write("\n")

    f.write("} // namespace qp_data\n")
