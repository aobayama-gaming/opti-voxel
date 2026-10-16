#pragma once

#include "godot_cpp/classes/node3d.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/variant.hpp"

#include "chunk_discoverer.hpp"

#include "godot_cpp/classes/camera3d.hpp"

#include "godot_cpp/classes/editor_interface.hpp"
#include "godot_cpp/classes/sub_viewport.hpp"

#include "mesher_pool.hpp"

#include <vector>

#include "chunk_memory.hpp"

#include "GPU_mesher.hpp"
#include "GPU_direct_mesher.hpp"
#include "mesh_validator.hpp"
#include "gpu_profiler.hpp"
#include "page_auditor.hpp"
#include "debug_log.hpp"

//debug
#include "sdf_dummy.hpp"

using namespace godot;




class VoxelTerrain : public Node3D {
	GDCLASS(VoxelTerrain, Node3D)

public:
	enum PipelineMode {
		PIPELINE_CPU = 0,
		PIPELINE_GPU_LEGACY = 1,
		PIPELINE_GPU_DIRECT = 2,
	};

protected:
	static void _bind_methods();

private:

    Camera3D *camera = nullptr;

	ChunkParam chunk_param{
		32,     // chunk_size
		0.5f,  // voxel_size
		2,      // skirt_size
		true    // use_skirt
	};

    ChunkDiscoverer* chunk_discoverer = nullptr;

	RID scenario;

	MesherPool* visual_mesh = nullptr;

	GPUMesher* gpu_mesher = nullptr;

	GPUDirectMesher* direct_mesher = nullptr;

	MeshValidator* validator = nullptr;
	GPUProfiler* gpu_profiler = nullptr;
	PageAuditor* auditor = nullptr;
	Dictionary audit_report;
	bool debug_gpu_profile = false;
	Dictionary validation_report;

	SDFBase* sdf = memnew(SDFDummy);

	std::vector<ChunkOutput *> visual_mesh_storage;

	VoxelState visual_mesh_data;

	// Settings (must be set before the node enters the tree).
	PipelineMode pipeline_mode = PIPELINE_GPU_LEGACY;
	int legacy_batch_size = 1024;
	int cpu_workers = 8;
	int lod_numbers = 15;
	bool legacy_trim_indices = false;
	float lod_pixel_error = 2.0f;
	bool debug_test_rig = false;
	GPUDirectMesher::Settings direct_settings;
	int debug_view = 0;

	bool started = false;

	void _destroy_chunk();

	void _teardown();

	//debug

	uint32_t exec_time = 0;
	bool scan_received = false;
	int64_t frames_until_loaded = -1;
	uint64_t start_usec = 0;
	int64_t usec_until_loaded = -1;

	struct ProcessStats {
		TimingStat process_total;
		TimingStat scan_ingest;      // get_scanned_chunks + add_scan_chunk + send_chunk
		TimingStat destroy_from_scan; // VoxelState::update_destroy_from_scan
		TimingStat mesher_dispatch;  // process_queue
		TimingStat mesher_collect;   // get_results
		TimingStat destroy_chunks;   // _destroy_chunk
	} process_stats;

public:

    void _ready() override;
    void _process(double delta) override;
	void _exit_tree() override;

	void set_pipeline_mode(int p_mode);
	int get_pipeline_mode() const;

	void set_legacy_batch_size(int p_size);
	int get_legacy_batch_size() const;

	// Screen-space error target of the LOD selection (higher = coarser = fewer chunks/triangles).
	// Edge length of a LOD 0 voxel in metres (each LOD level doubles it).
	void set_voxel_size(float p_size);
	float get_voxel_size() const;

	// Adds centimetre-scale test shapes in front of the main_scene spawn (see sdf_test_rig.hpp).
	void set_debug_test_rig(bool p_enable);
	bool get_debug_test_rig() const;

	void set_lod_pixel_error(float p_error);
	float get_lod_pixel_error() const;

	void set_legacy_trim_indices(bool p_trim);
	bool get_legacy_trim_indices() const;

	int get_pending_chunk_count() const;

	// Per-pass GPU timings in get_debug_stats()["gpu_breakdown"] (needs the viewport to measure its
	// render time, which is what makes the renderer capture timestamps).
	void set_debug_gpu_profile(bool p_enable);
	bool get_debug_gpu_profile() const;

	void set_direct_batch_size(int p_size);
	int get_direct_batch_size() const;
	void set_direct_pages_per_batch(int p_pages);
	int get_direct_pages_per_batch() const;
	void set_direct_page_capacity(int p_capacity);
	int get_direct_page_capacity() const;
	void set_direct_initial_pages(int p_pages);
	int get_direct_initial_pages() const;

	// GPU Direct only: 0 shaded, 1 normals, 2 lod, 3 page, 4 wireframe, 5 chunk.
	void set_debug_view(int p_view);
	int get_debug_view() const;

	void set_log_level(int p_level);
	int get_log_level() const;

	// True once every scanned chunk is meshed and the mesher has nothing in flight.
	bool is_load_complete();

	Dictionary get_debug_stats();

	// GPU Direct only: re-mesh `count` loaded chunks with the legacy mesher and compare triangle by
	// triangle. Asynchronous: poll get_validation_report() (empty until done). OBJ files of the first
	// mismatching chunk are written to `dump_dir` when it is not empty.
	bool debug_validate(int p_count, const String &p_dump_dir);
	Dictionary get_validation_report() const;

	// GPU Direct only: read back every page holding loaded chunks and classify all drawn quad slots
	// (live / dead / stale / partial). Asynchronous: poll get_audit_report().
	bool debug_audit_pages();
	Dictionary get_audit_report() const;

	VoxelTerrain() = default;
	~VoxelTerrain() override;

};

VARIANT_ENUM_CAST(VoxelTerrain::PipelineMode);
