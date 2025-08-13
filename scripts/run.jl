# This file sets up the necessary environment and calls the main functions to start the simulation.

using CUDA
using .Main

function run_simulation()
    # Initialize the simulation environment
    println("Initializing the black hole simulation...")

    # Set up the GPU environment
    CUDA.@init

    # Call the main function to start the simulation
    Main.main()
end

run_simulation()