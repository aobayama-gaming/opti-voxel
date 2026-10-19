// Reference runner for the Luau port: drives the real CPUMesher passes and SDFDummy (copied
// unchanged from src/ by build.sh) and prints their output.
//
//   oracle sdf < points            one "x y z" per line -> one SDF value per line
//   oracle mesh cx cy cz [skirt]   meshes chunk (cx, cy, cz) -> "V n", n "x y z nx ny nz", "T m", m "a b c"

#include <godot_shim.hpp>

#include <cstdlib>
#include <cstring>

#include "sdf_dummy.hpp"
#define private public
#include "CPU_mesher.hpp"
#undef private

int main(int argc, char **argv) {
	if (argc >= 2 && std::strcmp(argv[1], "sdf") == 0) {
		SDFDummy sdf;
		double x, y, z;
		while (std::scanf("%lf %lf %lf", &x, &y, &z) == 3) {
			std::printf("%.9g\n", (double)sdf.evaluate(Vector3(x, y, z)));
		}
		return 0;
	}

	if (argc >= 5 && std::strcmp(argv[1], "mesh") == 0) {
		SDFDummy sdf;
		CPUMesher mesher;
		mesher.sdf = &sdf;
		mesher.chunk_param = ChunkParam{ 32, 0.5f, 2, argc < 6 || std::atoi(argv[5]) != 0 };
		mesher.chunk_id = Vector3i(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
		mesher.per_face_normal = false;
		mesher.execute_passes();

		const VerticesData &d = mesher.vertices_data;
		const int64_t n = d.output_points.size();
		std::printf("V %lld\n", (long long)n);
		for (int64_t i = 0; i < n; i++) {
			const Vector3 &p = d.output_points[i];
			const Vector3 &nm = d.output_normals[i];
			std::printf("%.9g %.9g %.9g %.9g %.9g %.9g\n", (double)p.x, (double)p.y, (double)p.z, (double)nm.x, (double)nm.y, (double)nm.z);
		}
		// The index buffer is over-allocated (6 per crossing edge); unused quads stay all-zero.
		std::vector<int32_t> tris;
		for (int64_t i = 0; i + 2 < d.output_vertices.size(); i += 3) {
			const int32_t a = d.output_vertices[i], b = d.output_vertices[i + 1], c = d.output_vertices[i + 2];
			if (a == b || b == c || a == c) {
				continue;
			}
			tris.push_back(a);
			tris.push_back(b);
			tris.push_back(c);
		}
		std::printf("T %zu\n", tris.size() / 3);
		for (size_t i = 0; i < tris.size(); i += 3) {
			std::printf("%d %d %d\n", tris[i], tris[i + 1], tris[i + 2]);
		}
		return 0;
	}

	std::fprintf(stderr, "usage: oracle sdf < points | oracle mesh cx cy cz [skirt]\n");
	return 1;
}
