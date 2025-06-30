#include <iostream>
#include <vector>
#include <mpi.h>

/**
 * @brief A simple MPI "Hello World" program.
 *
 * This program initializes the MPI environment, determines its rank and the total
 * number of processes (world size), and prints a message from each process.
 * The root process (rank 0) also prints the version of the MPI library being used.
 *
 * To compile and run:
 * 1. Make sure you have an MPI implementation installed (e.g., OpenMPI, MPICH).
 * 2. Use the provided CMakeLists.txt to build the project.
 * 3. Run the executable using mpirun, e.g.:
 * mpirun -np 4 ./freeimpala_mpi
 */
int main(int argc, char** argv) {
    // Initialize the MPI environment. This must be the first MPI call.
    MPI_Init(&argc, &argv);

    // Get the total number of processes in the communicator.
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank (ID) of the current process. Ranks range from 0 to world_size-1.
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor (node) this process is running on.
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print a hello world message including the processor name, rank, and world size.
    std::cout << "Hello from processor '" << processor_name
              << "', rank " << world_rank << " out of " << world_size
              << " processes." << std::endl;

    // The root process (rank 0) is often used for special tasks like printing summary information.
    if (world_rank == 0) {
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        int version_len;
        MPI_Get_library_version(version, &version_len);
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "MPI library version: " << version << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // Finalize the MPI environment. This must be the last MPI call.
    MPI_Finalize();

    return 0;
}
