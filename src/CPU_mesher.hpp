#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include "chunk_math.hpp"
#include "lod_structures.hpp"
#include "sdf_base.hpp"
#include "chunk_memory.hpp"

using namespace godot;



namespace MeshEdgeUtils {
bool edge_change(uint32_t edge_case);
bool transversal_change(uint32_t origin_edge, uint32_t paired_edge, bool front);
uint32_t transversal_combination(uint32_t origin_edge, uint32_t paired_edge, bool front);
}

struct VerticesData {

    struct EdgeGrid3D {
        int32_t width = 0;
        int32_t height = 0;
        int32_t depth = 0;
        std::vector<uint32_t> values;

        void resize(int32_t p_width, int32_t p_height, int32_t p_depth) {
            width = p_width;
            height = p_height;
            depth = p_depth;
            values.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(depth), 0);
        }

        int32_t index(int32_t x, int32_t y, int32_t z) const {
            return x + width * (y + height * z);
        }

        std::vector<uint32_t>::reference operator()(int32_t x, int32_t y, int32_t z) {
            return values[static_cast<size_t>(index(x, y, z))];
        }

        std::vector<uint32_t>::reference operator()(const Vector3i &coord) {
            return (*this)(coord.x, coord.y, coord.z);
        }

        uint32_t operator()(int32_t x, int32_t y, int32_t z) const {
            return values[static_cast<size_t>(index(x, y, z))];
        }

        uint32_t operator()(const Vector3i &coord) const {
            return (*this)(coord.x, coord.y, coord.z);
        }

    };

    // Metadata grid of size (chunk_size +1)^2
    struct Metadata {

        int32_t x_start_trim = 0;
        int32_t x_end_trim = 0;

        int32_t start_trim = 0;
        int32_t end_trim = 0;

        struct Counts {
            int32_t x_edge = 0;
            int32_t y_edge = 0;
            int32_t z_edge = 0;
            int32_t point = 0;
        };

        Counts counts;

        //Counts inner_counts;
    };

    struct EdgeCompute {
        std::vector<Vector3> local_positions;
        std::vector<Vector3> normals;

        void configure(int32_t p_count) {
            local_positions.resize(static_cast<size_t>(p_count));
            normals.resize(static_cast<size_t>(p_count));
        }
    };

    struct MetadataGrid {
        int32_t height = 0;
        int32_t depth = 0;
        std::vector<Metadata> values;
        std::vector<Metadata::Counts> cumulative_counts;
        //std::vector<Metadata::Counts> inner_cumulative_counts;
        bool cache_valid = false;

        void resize(int32_t p_height, int32_t p_depth) {
            height = p_height;
            depth = p_depth;
            values.assign(static_cast<size_t>(height) * static_cast<size_t>(depth), Metadata {});
            cumulative_counts.assign(values.size(), Metadata::Counts {});
            //inner_cumulative_counts.assign(values.size(), Metadata::Counts {});
            cache_valid = false;
        }

        int32_t index(int32_t y, int32_t z) const {
            return z * height + y;
        }

        Metadata &operator()(int32_t y, int32_t z) {
            return values[static_cast<size_t>(index(y, z))];
        }

        const Metadata &operator()(int32_t y, int32_t z) const {
            return values[static_cast<size_t>(index(y, z))];
        }

        const Metadata::Counts &cum(int32_t y, int32_t z) const {
            return cumulative_counts[static_cast<size_t>(index(y, z))];
        }

        // const Metadata::Counts &int_cum(int32_t y, int32_t z) const {
        //     return inner_cumulative_counts[static_cast<size_t>(index(y, z))];
        // }

        void cache() {
            cumulative_counts.resize(values.size());
            //inner_cumulative_counts.resize(values.size());

            if (depth <= 0 || height <= 0 || values.empty()) {
                cache_valid = true;
                return;
            }

            Metadata::Counts row_accumulator {};
            //Metadata::Counts inner_row_accumulator {};
            
            for (int32_t z = 0; z < depth; ++z) {
                

                for (int32_t y = 0; y < height; ++y) {
                    const size_t i = static_cast<size_t>(index(y, z));
                    const Metadata &source = values[i];

                    row_accumulator.x_edge += source.counts.x_edge;
                    row_accumulator.y_edge += source.counts.y_edge;
                    row_accumulator.z_edge += source.counts.z_edge;
                    row_accumulator.point += source.counts.point;

                    cumulative_counts[i] = row_accumulator; // Stored after in order to have the total number directly, (we have last index of each y,z +1)
                    

                    // inner_row_accumulator.x_edge += source.inner_counts.x_edge;
                    // inner_row_accumulator.y_edge += source.inner_counts.y_edge;
                    // inner_row_accumulator.z_edge += source.inner_counts.z_edge;
                    // inner_row_accumulator.point += source.inner_counts.point;

                    // inner_cumulative_counts[i] = inner_row_accumulator; // Stored after in order to have the total number directly, (we have last index of each y,z +1)
                    

                }
            }

            cache_valid = true;
        }

        bool is_cached() const {
            return cache_valid;
        }
    };

    EdgeGrid3D x_edge_cases;                 // size = all cell corner (chunk_size +1)^3

    int32_t width;
    int32_t height;
    int32_t depth;

    MetadataGrid metadata;

    //result vector
    EdgeCompute x_edge;
    EdgeCompute y_edge;
    EdgeCompute z_edge;

    HashMap<Vector3i,int32_t> edge_cache;

    std::vector<Vector3> points;

    PackedVector3Array output_points;
    PackedVector3Array output_normals;
    PackedInt32Array output_vertices;

    void clear_all() {
        width = 0;
        height = 0;
        depth = 0;

        x_edge_cases.width = 0;
        x_edge_cases.height = 0;
        x_edge_cases.depth = 0;
        x_edge_cases.values.clear();

        metadata.height = 0;
        metadata.depth = 0;
        metadata.values.clear();
        metadata.cumulative_counts.clear();
        metadata.cache_valid = false;

        x_edge.local_positions.clear();
        x_edge.normals.clear();
        y_edge.local_positions.clear();
        y_edge.normals.clear();
        z_edge.local_positions.clear();
        z_edge.normals.clear();

        edge_cache.clear();
        points.clear();

        output_points.clear();
        output_normals.clear();
        output_vertices.clear();
    }

    bool cell_has_point(int32_t x, int32_t y, int32_t z);

    void configure(int32_t p_width, int32_t p_height, int32_t p_depth) { // The size of this should one unit bigger (difference between center and corner)
        
        clear_all();
        
        width = p_width;
        height = p_height;
        depth = p_depth;
        x_edge_cases.resize(p_width, p_height, p_depth);
        metadata.resize(p_height, p_depth);
    }

    void cache(bool per_face_normal=true)
    {
        metadata.cache();

        auto& max_meta = metadata.cum(height-1,depth-1);

        points.resize(max_meta.point);
        x_edge.configure(max_meta.x_edge);
        y_edge.configure(max_meta.y_edge);
        z_edge.configure(max_meta.z_edge);

        // DEPRECATED : over-allocate (because on edge we don t create the quad, actually don't care ? ) 

        output_vertices.resize((max_meta.x_edge+max_meta.y_edge+max_meta.z_edge)*6);

        if(per_face_normal){

            output_points.resize((max_meta.x_edge+max_meta.y_edge+max_meta.z_edge)*6);
            output_normals.resize((max_meta.x_edge+max_meta.y_edge+max_meta.z_edge)*6);

        }
        else{
            output_points.resize(max_meta.point);
            output_normals.resize(max_meta.point);
        }

    }

    bool vertices_outside(int x,int y, int z){

        return x==0 || x==depth-1 || y==0 ||y==height-1 || z==0 || z==depth-1 ;

    }
    
};


class CPUMesher : public Object {

private:

    //constant between run (no thread safe enforcement)
    Ref<Thread> worker;
    SDFBase* sdf;
    ChunkParam chunk_param;
    ChunkOutput chunk_output;
    RID scenario;

    bool per_face_normal = false;

    //worker availability
    Ref<Mutex> mutex;
    Ref<Semaphore> semaphore;
    bool worker_available = true;
    bool result_available = false;

    // Internal thread variable.
    VerticesData vertices_data;
    Vector3i chunk_id;

    void pipeline();

    float skirt_offset(float x,float y, float z, float value);
    float skirt_offset(Vector3 pos,float value);

    void _find_edge_intersection(const Vector3i &start_point, const Vector3i &end_point, const uint32_t edge_case, VerticesData::EdgeCompute &edge_data, uint32_t &index);

    void first_pass(const int32_t p_y_edge,const int32_t p_z_edge);
    void second_pass(const int32_t p_y_cell,const int32_t p_z_cell);
    void second_half_pass(const int32_t p_y_cell,const int32_t p_z_cell);
    void third_pass(const int32_t p_y_edge,const int32_t p_z_edge);
    void fourth_pass(const int32_t p_y_cell,const int32_t p_z_cell,const float_t alpha);
    void fifth_pass(const int32_t p_y_qcell,const int32_t p_z_qcell);

    void execute_passes();

    void render_chunk();

    void render_chunk_box();

public :

    bool is_available();

    void start(
        SDFBase* p_sdf,
        ChunkParam p_chunk_param,
        RID p_scenario
    );

    bool get_result(ChunkOutput &out);

    void launch_pipeline(Vector3i p_chunk_id);

};