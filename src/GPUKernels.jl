module GPUKernels

using CUDA, StaticArrays

export geodesic_kernel!

@inline function rhs(rs::Float32, E::Float32, y::NTuple{6,Float32})
    r, th, ph, dr, dth, dph = y
    f = 1f0 - rs/r
    dt = E / f
    sinth = sin(th); costh = cos(th)
    sinth = ifelse(sinth < 1f-6, 1f-6, sinth)
    invr = 1f0/r
    # second derivatives
    d2r  = - (rs /(2f0*r*r))*f*dt*dt + (rs/(2f0*r*r*f))*dr*dr + r*(dth*dth + sinth*sinth*dph*dph)
    d2th = -2f0*dr*dth*invr + sinth*costh*dph*dph
    d2ph = -2f0*dr*dph*invr - 2f0*costh/sinth * dth*dph
    return (dr, dth, dph, d2r, d2th, d2ph)
end

@inline function rk4(rs::Float32, E::Float32, h::Float32, y::NTuple{6,Float32})
    k1 = rhs(rs,E,y)
    y2 = ntuple(i-> y[i] + 0.5f0*h*k1[i], 6)
    k2 = rhs(rs,E,y2)
    y3 = ntuple(i-> y[i] + 0.5f0*h*k2[i], 6)
    k3 = rhs(rs,E,y3)
    y4 = ntuple(i-> y[i] + h*k3[i], 6)
    k4 = rhs(rs,E,y4)
    yN = ntuple(i-> y[i] + (h/6f0)*(k1[i] + 2f0*k2[i] + 2f0*k3[i] + k4[i]), 6)
    return yN
end

@inline function init_state(rs::Float32, cam_pos::SVector{3,Float32}, dir::SVector{3,Float32})
    r = sqrt(sum(abs2, cam_pos))
    invr = 1f0/r
    zc = cam_pos[3]*invr
    zc = clamp(zc, -1f0, 1f0)
    th = acos(zc)
    ph = atan(cam_pos[2], cam_pos[1])
    sinth = sin(th); sinth = sinth < 1f-6 ? 1f-6 : sinth
    # local spherical basis
    e_r  = SVector(sinth*cos(ph), sinth*sin(ph), cos(th))
    e_th = SVector(cos(th)*cos(ph), cos(th)*sin(ph), -sin(th))
    e_ph = SVector(-sin(ph), cos(ph), 0f0)
    dr  = Float32(dot(dir, e_r))
    dth = Float32(dot(dir, e_th) * invr)
    dph = Float32(dot(dir, e_ph) / (r*sinth))
    f = 1f0 - rs/r
    rad = (dr*dr)/f + (r*r)*(dth*dth + sinth*sinth*dph*dph)
    rad = rad < 0f0 ? 0f0 : rad
    dt_dλ = sqrt(rad)
    E = f * dt_dλ
    return (Float32(r), Float32(th), Float32(ph), dr, dth, dph), Float32(E)
end

function geodesic_kernel!(rs::Float32, cam_pos::SVector{3,Float32}, forward::SVector{3,Float32}, right::SVector{3,Float32}, up::SVector{3,Float32}, fovY::Float32, aspect::Float32,
                          cx,cy,cz,radius,color,nobj::Int32,
                          disk_r1::Float32, disk_r2::Float32,
                          hit_type, hit_idx, hit_x, hit_y, hit_z,
                          col_r, col_g, col_b, col_a,
                          width::Int32, height::Int32, maxSteps::Int32, h::Float32, escape_r::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i >= width || j >= height
        return
    end
    # pixel to NDC
    fx = (Float32(i) + 0.5f0)/Float32(width)
    fy = (Float32(j) + 0.5f0)/Float32(height)
    ndc_x = 2f0*fx - 1f0
    ndc_y = 1f0 - 2f0*fy
    tanHalf = tan(0.5f0 * fovY)
    dir_cam = SVector(ndc_x * tanHalf * aspect, ndc_y * tanHalf, -1f0)
    # build world dir
    dir_world = dir_cam[1]*right + dir_cam[2]*up + dir_cam[3]*forward
    # normalize
    invlen = rsqrt(sum(abs2, dir_world))
    dir_world = dir_world * invlen
    state, E = init_state(rs, cam_pos, dir_world)
    r, th, ph, dr, dth, dph = state
    hitcode::UInt8 = 0x00  # 0 miss,1 horizon,2 disk,3 object
    objhit::Int32 = -1
    x = 0f0; y = 0f0; z = 0f0
    prev_y = cam_pos_f32 = cam_pos[2]  # previous y for disk crossing
    center_hit_x = 0f0; center_hit_y = 0f0; center_hit_z = 0f0; center_hit_r = 0f0
    for step in 1:maxSteps
        # cartesian position
        sinth = sin(th); costh = cos(th)
        x = r * sinth * cos(ph)
        y = r * sinth * sin(ph)
        z = r * costh
        if r <= rs + 1e-4f0*rs
            hitcode = 0x01
            break
        end
        if r > escape_r
            hitcode = 0x00
            break
        end
        # sphere objects
        # disk intersection (equatorial plane y=0). Using sign change of y (previous vs current) and radial bounds in xz-plane.
        if (prev_y * y < 0f0)
            r_disk = sqrt(x*x + z*z)
            if (r_disk >= disk_r1 && r_disk <= disk_r2)
                hitcode = 0x02
                break
            end
        end
        prev_y = y

        # objects (spheres)
        if hitcode == 0x00 && nobj > 0
            @inbounds for k in 1:nobj
                dx = x - cx[k]; dy = y - cy[k]; dz = z - cz[k]
                if (dx*dx + dy*dy + dz*dz) <= radius[k]*radius[k]
                    hitcode = 0x03
                    objhit = k-1
                    center_hit_x = cx[k]; center_hit_y = cy[k]; center_hit_z = cz[k]; center_hit_r = radius[k]
                    break
                end
            end
            if hitcode == 0x03
                break
            end
        end
        # advance
        state = rk4(rs, E, h, (r,th,ph,dr,dth,dph))
        r, th, ph, dr, dth, dph = state
    end
    idx = j*width + i + 1  # 1-based indexing for Julia arrays
    hit_type[idx] = hitcode
    hit_idx[idx] = objhit
    hit_x[idx] = x
    hit_y[idx] = y
    hit_z[idx] = z
    # Color logic mirroring original GLSL shader
    cr = 0f0; cg = 0f0; cb = 0f0; ca = 0f0
    if hitcode == 0x01
        cr = 0f0; cg = 0f0; cb = 0f0; ca = 1f0
    elseif hitcode == 0x02
        r_rel = sqrt(x*x + z*z) / disk_r2
        cr = 1f0; cg = r_rel; cb = 0.2f0; ca = r_rel
    elseif hitcode == 0x03
        # shading: V = cam_pos - P; N = normalize(P - center)
        Vx = cam_pos[1] - x; Vy = cam_pos[2] - y; Vz = cam_pos[3] - z
        invVL = rsqrt(Vx*Vx + Vy*Vy + Vz*Vz)
        Vx*=invVL; Vy*=invVL; Vz*=invVL
        Nx = x - center_hit_x; Ny = y - center_hit_y; Nz = z - center_hit_z
        invNL = rsqrt(Nx*Nx + Ny*Ny + Nz*Nz)
        Nx*=invNL; Ny*=invNL; Nz*=invNL
        diff = Nx*Vx + Ny*Vy + Nz*Vz
        if diff < 0f0; diff = 0f0; end
        ambient = 0.1f0
        intensity = ambient + (1f0-ambient)*diff
        k = objhit + 1
        base_r = color[1,k]; base_g = color[2,k]; base_b = color[3,k]; base_a = color[4,k]
        cr = base_r * intensity; cg = base_g * intensity; cb = base_b * intensity; ca = base_a
    end
    col_r[idx] = cr; col_g[idx] = cg; col_b[idx] = cb; col_a[idx] = ca
    return
end

end # module