
def format_bytes(bit_count):
    byte_count = bit_count / 8
    if byte_count >= 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.2f} MiB"
    if byte_count >= 1024:
        return f"{byte_count / 1024:.2f} KiB"
    return f"{byte_count:.2f} B"


def print_memory_footprint(memory_footprint, chunk_number, title="Memory footprint"):
    total_bits_per_chunk = sum(memory_footprint.values())
    total_bits = total_bits_per_chunk * chunk_number

    print(f"{title} for {chunk_number} chunks")
    print(f"Per chunk total: {format_bytes(total_bits_per_chunk)}")
    print(f"All chunks total: {format_bytes(total_bits)}")
    print()
    print("Breakdown:")

    for name, bit_count in sorted(memory_footprint.items(), key=lambda item: item[1], reverse=True):
        percentage = bit_count / total_bits_per_chunk * 100 if total_bits_per_chunk else 0
        print(f"- {name}: {format_bytes(bit_count)} ({percentage:.2f}% of one chunk)")


def print_global_footprint(global_footprint, title="Global footprint"):
    total_bits = sum(global_footprint.values())

    print(title)
    print(f"Total: {format_bytes(total_bits)}")
    print()
    print("Breakdown:")

    for name, bit_count in sorted(global_footprint.items(), key=lambda item: item[1], reverse=True):
        percentage = bit_count / total_bits * 100 if total_bits else 0
        print(f"- {name}: {format_bytes(bit_count)} ({percentage:.2f}% of global)")


if __name__=="__main__":

    chunk_size = 32

    chunk_number = 100

    # print(" # NAIVE WAY")

    # float_precision = 64

    # int_precision = 32

    # # memory footprint in bits

    # memory_footprint = {}

    # memory_footprint["sdf_sample"] = (chunk_size+1)**3*float_precision
    # memory_footprint["out_point"] = chunk_size**3*3*float_precision
    # memory_footprint["out_normal"] = chunk_size**3*3*float_precision
    # memory_footprint["out_vertex"] = 6*3*(chunk_size-1)**3*int_precision
    # memory_footprint["edge_intersect"] = float_precision*3*(chunk_size-1)*chunk_size**2

    # print_memory_footprint(memory_footprint, chunk_number, title="Memory footprint")

    print(" # BRUTE FORCE / SPARSE")

    bool_precision = 1
    uint8_precision = 8
    uint16_precision = 16
    uint32_precision = 32

    global_footprint = {}
    global_footprint["global_point_atomic_counter"] = uint32_precision
    global_footprint["global_triangle_atomic_counter"] = uint32_precision

    print_global_footprint(global_footprint, title="Global footprint")

    memory_footprint = {}

    # Pass 1 (Vertex): Determine the sdf sample.
    memory_footprint["sdf_sample"] = (chunk_size + 1)**3 * uint32_precision *3 

    # Pass 2, 3, 4 (Edge): Determine X/Y/Z edge crossing.
    memory_footprint["edge_intersect"] = 3 * (chunk_size - 1) * chunk_size**2 * uint32_precision

    # Per-chunk counters.
    memory_footprint["per_chunk_point_atomic_counter"] = uint32_precision
    memory_footprint["per_chunk_triangle_atomic_counter"] = uint32_precision

    memory_footprint["point_map"] = chunk_size**3*uint32_precision

    # Pass 5 (Cell): QEF / point - normals. + Quad
    memory_footprint["output_point"] = chunk_size**3 * 3 * uint8_precision
    memory_footprint["output_normal"] = chunk_size**3 * 3 * uint8_precision
    memory_footprint["output_point_metadata"] = chunk_size**3 * (uint32_precision + uint16_precision)


    print_memory_footprint(memory_footprint, chunk_number, title="Memory footprint")

    # print(" # OPTIMIZED")

    # #reduce for normal precision
    # float_precision = 32

    # # chunk_size is 32 which means we need 32,768 index which is 8bit so theorically can pack it in 8bit uint inside
    # int_precision = 8

    # # memory footprint in bits

    # memory_footprint = {}

    # memory_footprint["sdf_sample"] = (chunk_size+1)**3*float_precision
    # memory_footprint["out_point"] = chunk_size**3*3*float_precision

    # # Compress normal in one float
    # memory_footprint["out_normal"] = chunk_size**3*float_precision
    # memory_footprint["out_vertex"] = 6*3*(chunk_size-1)**3*int_precision

    # # only need to store the edge intersect on a uint8 (to get the proportion since we are already doing binary search)
    # memory_footprint["edge_intersect"] = int_precision*3*(chunk_size-1)*chunk_size**2

    # print_memory_footprint(memory_footprint, chunk_number, title="Memory footprint")

    