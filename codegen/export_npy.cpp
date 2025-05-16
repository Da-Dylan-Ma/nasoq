#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdint>
#include "../examples/yaml_qp_parser.h"

// Write raw array as .npy format (little-endian, no compression)
template<typename T>
void write_npy(const std::string &filename, const T *data, int64_t len) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    std::string header =
            "{'descr': '" + std::string((sizeof(T) == 8 ? "<f8" : "<i4")) +
            "', 'fortran_order': False, 'shape': (" + std::to_string(len) + ",), }";

    size_t pad_len = 16 - (10 + header.size() + 1) % 16;
    header += std::string(pad_len, ' ');
    header += '\n';

    fout.write("\x93NUMPY", 6);
    fout.put(0x01); fout.put(0x00);
    uint16_t hlen = static_cast<uint16_t>(header.size());
    fout.write(reinterpret_cast<char *>(&hlen), 2);
    fout.write(header.c_str(), header.size());
    fout.write(reinterpret_cast<const char *>(data), sizeof(T) * len);
    fout.close();
}

void export_csc(const nasoq::CSC *M, const std::string &prefix) {
    int ncol = M->ncol;
    int nnz  = M->p[ncol];

    write_npy(prefix + "_p.npy", M->p, ncol + 1);
    write_npy(prefix + "_i.npy", M->i, nnz);
    write_npy(prefix + "_x.npy", M->x, nnz);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./export_npy <qp_file.yml>\n";
        return 1;
    }

    std::string fname = argv[1];
    QPProblem qp;
    if (!parse_qp_yaml(fname, qp)) {
        std::cerr << "Failed to parse the .yml file.\n";
        return 2;
    }

    std::cout << "Exporting parsed QP...\n";

    export_csc(qp.H, "H_csc");
    write_npy("q.npy", qp.q, qp.n);

    if (qp.me > 0 && qp.A && qp.b) {
        export_csc(qp.A, "A_csc");
        write_npy("b.npy", qp.b, qp.me);
    }

    if (qp.mi > 0 && qp.C && qp.l && qp.u) {
        export_csc(qp.C, "C_csc");
        write_npy("l.npy", qp.l, qp.mi);
        write_npy("u.npy", qp.u, qp.mi);
    }

    std::cout << "Output written as .npy files.\n";
    return 0;
}
