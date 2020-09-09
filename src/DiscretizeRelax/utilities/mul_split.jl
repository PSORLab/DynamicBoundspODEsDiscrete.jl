function mul_split!(Y::Vector{R}, A::Matrix{S}, B::Vector{T}, nx) where {R,S,T}
    if nx == 1
        @inbounds Y[1] = A[1,1]*B[1]
    else
        mul!(Y, A, B)
    end

    return nothing
end

function mul_split!(Y::Matrix{R}, A::Matrix{S}, B::Matrix{T}, nx) where {R,S,T}
    if nx == 1
        @inbounds Y[1,1] = A[1,1]*B[1,1]
    else
        mul!(Y, A, B)
    end

    return nothing
end
