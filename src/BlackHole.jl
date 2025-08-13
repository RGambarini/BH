module BlackHole

using StaticArrays

export BlackHole, schwarzschild_radius, intercept

const G = 6.67430e-11
const c = 2.99792458e8

schwarzschild_radius(mass::Float64) = 2.0 * G * mass / c^2

struct BlackHole
    position::SVector{3,Float64}
    mass::Float64
    r_s::Float64
end

BlackHole(pos::AbstractVector{<:Real}, m::Real) = begin
    p = SVector{3,Float64}(pos[1], pos[2], pos[3])
    rs = schwarzschild_radius(Float64(m))
    BlackHole(p, Float64(m), rs)
end

@inline function intercept(bh::BlackHole, p::SVector{3,Float64})
    return sum(abs2, p .- bh.position) <= bh.r_s^2
end

@inline function intercept(bh::BlackHole, px::Float64, py::Float64, pz::Float64)
    return intercept(bh, SVector(px,py,pz))
end

end # module