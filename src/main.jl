# Minimal driver producing positional hit data (no rendering loop)

using CUDA, StaticArrays
using .BlackHole
using .Camera
using .Objects
using .GPUKernels

function build_scene()
    bh = BlackHole([0.0,0.0,0.0], 8.54e36)
    objs = [
        ObjectData((4e11, 0.0, 0.0), 4e10, (1.0,1.0,0.0,1.0), 1.98892e30),
        ObjectData((0.0, 0.0, 4e11), 4e10, (1.0,0.0,0.0,1.0), 1.98892e30)
    ]
    return bh, objs
end

function launch_geodesics(;width=512, height=512, maxSteps=60_000, step_scale=1e7f0/6.0f0, escape_factor=1e20f0,
    disk_r1_factor=2.2, disk_r2_factor=5.2)
    bh, objs = build_scene()
    cam = CameraState()
    cam_pos, forward, right, up = camera_basis(cam)
    aspect = Float32(width/height)
    fovY = Float32(cam.fovY)
    rs = Float32(bh.r_s)
    # Convert vectors to Float32 SVector
    cam_pos_f = SVector{3,Float32}(Float32.(cam_pos))
    forward_f = SVector{3,Float32}(Float32.(forward))
    right_f = SVector{3,Float32}(Float32.(right))
    up_f = SVector{3,Float32}(Float32.(up))
    soa = to_device_soa(objs)
    Npix = width*height
    hit_type = CuArray{UInt8}(undef, Npix)
    hit_idx  = CuArray{Int32}(undef, Npix)
    hit_x = CuArray{Float32}(undef, Npix)
    hit_y = CuArray{Float32}(undef, Npix)
    hit_z = CuArray{Float32}(undef, Npix)
    col_r = CuArray{Float32}(undef, Npix)
    col_g = CuArray{Float32}(undef, Npix)
    col_b = CuArray{Float32}(undef, Npix)
    col_a = CuArray{Float32}(undef, Npix)
    h = step_scale * rs
    escape_r = escape_factor * rs
    threads = (16,16)
    blocks = (cld(width, threads[1]), cld(height, threads[2]))
    disk_r1 = Float32(disk_r1_factor * bh.r_s)
    disk_r2 = Float32(disk_r2_factor * bh.r_s)
    @cuda threads=threads blocks=blocks geodesic_kernel!(rs, cam_pos_f, forward_f, right_f, up_f, fovY, aspect,
        soa.cx, soa.cy, soa.cz, soa.radius, soa.color, Int32(soa.n),
        disk_r1, disk_r2,
        hit_type, hit_idx, hit_x, hit_y, hit_z,
        col_r, col_g, col_b, col_a,
        Int32(width), Int32(height), Int32(maxSteps), h, escape_r)
    synchronize()
    return (;hit_type=Array(hit_type), hit_idx=Array(hit_idx), hit_x=Array(hit_x), hit_y=Array(hit_y), hit_z=Array(hit_z),
            col_r=Array(col_r), col_g=Array(col_g), col_b=Array(col_b), col_a=Array(col_a), rs=bh.r_s,
            width, height)
end

# Note: not auto-running to avoid GPU dependency in non-GPU environments.
