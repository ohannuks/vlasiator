#include "gpu_velocity_grid.hpp"

using namespace spatial_cell;

int main(void) {
    init_spatial_cell_static();
    SpatialCell cell;
    
    const int ids_len = 5;
    int ids[] = {445566, 334499, 775555, 668844, 445511};

    // Add blocks to the given ids
    for (int i=0; i<ids_len; i++) {
        int ind = ids[i];
        cell.add_velocity_block(ind);
        Velocity_Block* block_ptr = cell.at(ind);
        block_ptr->data[0]=ind; // Put some data into each velocity cell
    }
    // Print data as it is on CPU
    printf("On host:\n");
    print_blocks(&cell);
    
    // Create a new instance. Constructor copies related data.
    GPU_velocity_grid *ggrid = new GPU_velocity_grid(&cell);
    

    // Print data from GPU
    printf("On GPU:\n");
    ggrid->print_blocks();
    printf("Print from kernel:\n");
    ggrid->k_print_blocks();
    
    print_constants();
    printf("%u %u %u\n", SpatialCell::vx_length, SpatialCell::vy_length, SpatialCell::vz_length);
    
    unsigned int min_ind = ggrid->min_ind(ids_len);
    printf("Min ind: %u\n", min_ind);
    ind3d min_indices = GPU_velocity_grid::get_velocity_block_indices_host(min_ind);
    printf("Min ind: %u (%u %u %u)\n", min_ind, min_indices.x, min_indices.y, min_indices.z);
    
    unsigned int max_ind = ggrid->max_ind(ids_len);
    printf("Max ind: %u\n", max_ind);
    ind3d max_indices = GPU_velocity_grid::get_velocity_block_indices_host(max_ind);
    printf("Max ind: %u (%u %u %u)\n", max_ind, max_indices.x, max_indices.y, max_indices.z);
    
    return 0;
}
