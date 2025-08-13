module Objects

using StaticArrays, CUDA

export ObjectData, update_position!, to_device_soa

struct ObjectData
    position::SVector{3,Float32}
    radius::Float32
    color::SVector{4,Float32}
    mass::Float32
    velocity::SVector{3,Float32}
end

ObjectData(pos::AbstractVector{<:Real}, radius::Real, color::AbstractVector{<:Real}, mass::Real; velocity = (0,0,0)) =
    ObjectData(SVector{3,Float32}(Float32.(pos)...), Float32(radius), SVector{4,Float32}(Float32.(color)...), Float32(mass), SVector{3,Float32}(Float32.(velocity)...))

function update_position!(obj::ObjectData, dt::Float32)
    obj = ObjectData(obj.position + obj.velocity*dt, obj.radius, obj.color, obj.mass; velocity=obj.velocity)
    return obj
end

"""Convert array of ObjectData to device SoA (structure of arrays)."""
function to_device_soa(objs::Vector{ObjectData})
    n = length(objs)
    cx = CuArray([o.position[1] for o in objs])
    cy = CuArray([o.position[2] for o in objs])
    cz = CuArray([o.position[3] for o in objs])
    radius = CuArray([o.radius for o in objs])
    color = CuArray(reshape(collect(Iterators.flatten((Tuple(o.color) for o in objs))), 4, n))
    return (;cx,cy,cz,radius,color,n)
end

end # module

