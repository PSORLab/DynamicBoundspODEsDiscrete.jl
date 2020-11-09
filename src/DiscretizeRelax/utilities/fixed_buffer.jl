mutable struct FixedCircularBuffer{T} <: AbstractVector{T}
    capacity::Int
    first::Int
    length::Int
    buffer::Vector{T}

    FixedCircularBuffer{T}(capacity::Int) where {T} = new{T}(capacity, 1, 0, Vector{T}(undef, capacity))
end

Base.@propagate_inbounds function _buffer_index_checked(cb::FixedCircularBuffer, i::Int)
    @boundscheck if i < 1 || i > cb.length
        throw(BoundsError(cb, i))
    end
    _buffer_index(cb, i)
end

@inline function _buffer_index(cb::FixedCircularBuffer, i::Int)
    n = cb.capacity
    idx = cb.first + i - 1
    return ifelse(idx > n, idx - n, idx)
end

@inline Base.@propagate_inbounds function Base.setindex!(cb::FixedCircularBuffer, data, i::Int)
    cb.buffer[_buffer_index_checked(cb, i)] = data
    return nothing
end

Base.size(cb::FixedCircularBuffer) = (length(cb),)
Base.convert(::Type{Array}, cb::FixedCircularBuffer{T}) where {T} = T[x for x in cb]

@inline Base.@propagate_inbounds function Base.getindex(cb::FixedCircularBuffer, i::Int)
    cb.buffer[_buffer_index_checked(cb, i)]
end

Base.length(cb::FixedCircularBuffer) = cb.length
Base.eltype(::Type{FixedCircularBuffer{T}}) where T = T

@inline function Base.push!(cb::FixedCircularBuffer, data)
    if cb.length == cb.capacity
        cb.first = (cb.first == cb.capacity ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    @inbounds cb.buffer[_buffer_index(cb, cb.length)] = data
    return cb
end

function Base.append!(cb::FixedCircularBuffer, datavec::AbstractVector)
    # push at most last `capacity` items
    n = length(datavec)
    for i in max(1, n-cb.capacity+1):n
        push!(cb, datavec[i])
    end
    return cb
end

function cycle!(cb::FixedCircularBuffer{S}) where S
    cb.first = (cb.first == 1 ? cb.capacity : cb.first - 1)
    return nothing
end

function cycle_copyto!(cb::FixedCircularBuffer{V}, v) where V <: AbstractArray
    cycle!(cb)
    copyto!(cb.buffer[1], v)
    return nothing
end

function cycle_eval!(f!, cb::FixedCircularBuffer{S}, x, p, t) where S
    cycle!(cb)
    f!(cb.buffer[1], x, p, t)
    return nothing
end
