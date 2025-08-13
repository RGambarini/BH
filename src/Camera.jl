module Camera

using StaticArrays
export CameraState, camera_position, camera_basis

struct CameraState
    target::SVector{3,Float64}
    radius::Float64
    azimuth::Float64
    elevation::Float64
    fovY::Float64
end

CameraState(;target = SVector(0.0,0.0,0.0), radius=6.34194e10, azimuth=0.0, elevation=π/2, fovY=45.0 * π/180) =
    CameraState(target, radius, azimuth, elevation, fovY)

@inline function camera_position(cam::CameraState)
    el = clamp(cam.elevation, 1e-3, π-1e-3)
    return SVector(
        cam.radius * sin(el) * cos(cam.azimuth),
        cam.radius * cos(el),
        cam.radius * sin(el) * sin(cam.azimuth)
    )
end

@inline function camera_basis(cam::CameraState)
    pos = camera_position(cam)
    forward = normalize(cam.target - pos)
    approx_up = SVector(0.0,1.0,0.0)
    right = normalize(cross(forward, approx_up))
    up = normalize(cross(right, forward))
    return pos, forward, right, up
end

end # module