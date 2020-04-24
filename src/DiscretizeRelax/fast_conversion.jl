# setindex! that avoids conversion.
function setindex!(x::Vector{STaylor1{N,T}}, val::T, i::Int) where {N,T}
    x[i] = STaylor1(val, Val{N-1}())
    nothing
end
