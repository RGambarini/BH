using Test
include("../src/BlackHole.jl")
include("../src/Camera.jl")
include("../src/Objects.jl")
include("../src/GPUKernels.jl")
include("../src/Renderer.jl")

# Test BlackHole functionality
function test_black_hole()
    bh = BlackHole(vec3(0.0, 0.0, 0.0), 8.54e36)
    @test bh.position == vec3(0.0, 0.0, 0.0)
    @test bh.mass == 8.54e36
    @test bh.Intercept(0.0, 0.0, 0.0) == true
    @test bh.Intercept(1e11, 1e11, 1e11) == false
end

# Test Camera functionality
function test_camera()
    cam = Camera()
    cam.processMouseMove(100.0, 200.0)
    @test cam.azimuth ≈ 1.0
    @test cam.elevation ≈ 1.0
    cam.processScroll(0.0, -1.0)
    @test cam.radius ≈ cam.minRadius
end

# Test ObjectData functionality
function test_object_data()
    obj = ObjectData(vec4(1.0, 2.0, 3.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), 1.0)
    @test obj.posRadius == vec4(1.0, 2.0, 3.0, 1.0)
    @test obj.color == vec4(1.0, 0.0, 0.0, 1.0)
    @test obj.mass == 1.0
end

# Test GPU Kernels functionality
function test_gpu_kernels()
    # Placeholder for GPU kernel tests
    # You can implement specific tests for your CUDA kernels here
    @test true
end

# Test Renderer functionality
function test_renderer()
    renderer = Renderer()
    @test renderer != nothing
    # Add more specific tests for rendering functionality
end

# Run all tests
function run_tests()
    test_black_hole()
    test_camera()
    test_object_data()
    test_gpu_kernels()
    test_renderer()
end

run_tests()