abstract type AbstractLinearMethod end
struct AdamsMoulton <: AbstractLinearMethod end
struct BDF <: AbstractLinearMethod end

"""
$(TYPEDEF)

A structure which holds an N-step parametric linear method of

$(TYPEDFIELDS)
"""
struct PLMS{N,T<:AbstractLinearMethod}
    "Time ordered from current time step 1 to (N-1)th prior time"
    times::CircularBuffer{Float64}
    "Coefficients of the PLMs method"
    coeffs::Vector{Float64}
    "A vector of length N used to compute β part coefficients"
    β::Vector{Vector{Float64}}
    "An upper triangular matrix used to compute g part of coefficients"
    c::Matrix{Float64}
    "A vector of length N+1 used to store the gcomponent of the coefficients"
    g::Vector{Float64}
    φ::Vector{Vector{NTuple{N,Float64}}}
end
function PLMS(x::Val{N}, s::T) where {N, T<:AbstractLinearMethod}
    c = CircularBuffer{Float64}(N)
    fill!(c, 0.0)
    β = Vector{Float64}[]
    for i in 1:N
        push!(β, zeros(Float64,N))
    end
    φ = Vector{NTuple{N,Float64}}[]
    for i in 1:N
        φin = NTuple{N,Float64}[]
        for j in 1:N
            push!(φin, ntuple(x -> 0.0, N))
        end
        push!(φ, φin)
    end
    PLMS{N,T}(c, zeros(Float64,N), β, zeros(Float64,N,N), zeros(Float64,N), φ)
end

@generated function seed_tuple(u::Val{U}, n::Val{N}) where {U,N}
    expr = Expr(:tuple)
    expr.args = [i == U ? 1.0 : 0.0 for i in 1:N]
    return expr
end

"""
$(TYPEDSIGNATURES)

Computes the cofficients of the N-step Adams Moulton method from the cofficients.
"""
function compute_coefficients!(x::PLMS{N, AdamsMoulton}) where N
    t = x.times

    # set β & last step accordingly
    fill!(x.β[1], 1.0)
    for i in 2:N
        @__dot__ x.β[i] = x.β[i-1]*(t[1] - t[i+1])/(t[2] - t[i+2])
    end
    hn = (t[1] - t[2])

    # set g
    for j = 0:(N-1)
        for q = 1:N
            if j == 0
                @inbounds x.c[1,q] = 1.0/q
            elseif j >= q
                @inbounds x.c[j+1, q] = x.c[j, q] - x.c[j,q+1]*hn/(t[1] - t[j+1])
            else
                break
            end
        end
        @inbounds x.g[j+1] = x.c[j+1,1]
    end

    # set φ
    map!(j -> seed_tuple(Val(j),Val(N)), x.φ[1], 1:N)
    for j=2:N
        for i=1:N
            if i <= j
                x.φ[j][i] = x.φ[j-1][i] - x.β[j-1][i+1]*x.φ[j-1][i+1]
            end
        end
    end

    # finish computing variable coefficients
    fill!(x.coeffs, 0.0)
    for i=1:N
        @__dot__ x.coeffs .+= (x.g[i]*x.β[i])*x.φ[i]
    end
    x.coeffs .*= hn

    nothing
end

"""
$(TYPEDEF)

Functor used to evaluate an N-step PLM method.
"""
struct PLMsFunctor{F,N,T,S,JX,JP}
    "PLMS Storage"
    plms::PLMS{N,T}
    "Circular Buffer for Holding Jx's"
    buffer_Jx::CircularBuffer{Matrix{S}}
    "Circular Buffer for Holding Jp's"
    buffer_Jp::CircularBuffer{Matrix{S}}
    "Circular Buffer for state bounds/relaxations X's"
    X::CircularBuffer{Vector{S}}
    "Circular Buffer for reference x"
    refx::CircularBuffer{Vector{Float64}}
    "Circular Buffer for times"
    t::CircularBuffer{Float64}
    "Holds parameter values P"
    P::Vector{S}
    "Holds parameter values shift by reference, P - p"
    rP::Vector{S}
    "Holds reference parameter value"
    p::Vector{Float64}
    "Number of state variables"
    nx::Int
    "Number of decision variables"
    np::Int
    "Rhs function"
    f!::F
    "Jacobian of the rhs w.r.t to the state variables"
    Jx!::JX
    "Jacobian of the rhs w.r.t to the descision variables"
    Jp!::JP
    "Temporary Storage for sum of the Jacobian w.r.t p "
    sJp::Matrix{S}
    δₖ
    Z
    x0
    x
    "Temporary storage Float64, nx-1"
    M1x::Vector{Float64}
end
function PLMsFunctor(s::S, plms::PLMS{N,T}, f!::F, Jx!::JX, Jp!::JP,
                     nx::Int, np::Int) where {F, JX, JP, S, N, T<:AbstractLinearMethod}
    buffer_Jx = CircularBuffer{Matrix{S}}(N)
    buffer_Jp = CircularBuffer{Matrix{S}}(N)
    X = CircularBuffer{Vector{S}}(N)
    t = CircularBuffer{Float64}(N)
    fill!(buffer_Jx, zeros(S, nx, nx))
    fill!(buffer_Jp, zeros(S, nx, np))
    fill!(X, zeros(S, nx))
    fill!(t, 0.0)
    P = zeros(S,np)
    rP = zeros(S,np)
    sJp = zeros(S, nx, np)
    δₖ = zeros(Float64, nx)
    Z = zeros(Float64, nx)
    x0 = zeros(Float64, nx)
    x = zeros(Float64, nx)
    M1x = zeros(Float64,nx)
    PLMsFunctor{N,T,S,JX,JP}(plms, buffer_Jx, buffer_Jp, X,
                             t, P, rP, nx, np, f!, Jx!, Jp!, sJp, δₖ,
                             Z, x0, x, M1x)
end


function update_coefficients!(pf::PLMsFunctor{F,N,T,S,JX,JP}, t0::Float64)
    pushfirst!(pf.plms.times, t0)
    compute_coefficients!(x.plms)
    nothing
end

function set_cycle_X!(pf::PLMsFunctor, Yⱼ)
    pf.X.first = (pf.X.first == 1 ? pf.X.length : pf.X.first - 1)
    copyto!(pf.X.buffer[pf.X.first], 1, Yⱼ, 1, pf.nx)
    nothing
end
eval_cycle_Jx!(pf::PLMsFunctor) = eval_cycle!(pf.Jx!, pf.buffer_Jx, first(pf.X), pf.P, first(pf.t))
eval_cycle_Jp!(pf::PLMsFunctor) = eval_cycle!(pf.Jp!, pf.buffer_Jp, first(pf.X), pf.P, first(pf.t))

function compute_sum_Jp!(pf::PLMsFunctor)
    map!((x,y) -> x.*y, pf.buffer_Jp, pf.plms.coeffs, pf.buffer_Jp)
    accumulate!(.+, pf.sJp, pf.buffer_Jp)
    nothing
end

function compute_δₖ!(pf::PILMsFunctor{N,T,S,JX,JP}) where {N,T,S,JX,JP}
    @__dot__ pf.δₖ = pf.x0 - pf.x + pf.R
    for i=1:N
        pf.f!(pf.M1x, pf.refx[i], pf.p, pf.t[i])
        @__dot__ pf.δₖ += pf.plms.coeffs[i]*pf.M1x
    end
    nothing
end

function refine_X!(pf::PILMsFunctor, A, Δⱼ)
    mul!(pf.M1x, pf.sJp, pf.rP)
    JxX1 = (pf.buffer_Jx[1] - mid(pf.buffer_Jx[1]))*(pf.X0 - pf.x0)
    JxX2 = ((I - pf.buffer_Jx[2])*A[1])*Δⱼ[1]
    @__dot__ pf.X = pf.x0 + pf.δₖ + pf.M2p + JxX1 + JxX2

    for i=2:N
        mul!(pf.M2x, pf.buffer_Jx[i], A[2])
        @__dot__ pf.X += pf.M2x*Δⱼ[2]
    end
    @__dot__ pf.X = pf.X ∩ pf.X0
    nothing
end

function compute_Δₖ!(pf::PILMsFunctor, A, Δⱼ)
    Yinv = inv((I - mid(pf.buffer_Jx[1]))*Aₖ)
    mul!(pf.M2xp, Yinv, pf.sJp)
    mul!(pf.M1x, pf.M2xp, pf.rP) # Yinv*Jkp*pf.rP
    mul!(pf.M1x_a, Yinv, pf.δₖ)

    JxX1 = Yinv*(pf.buffer_Jx[1] - mid(pf.buffer_Jx[1]))*(pf.X - pf.x)
    JxX2 = ((Yinv*(I - pf.buffer_Jx[1]))*A[1])*Δⱼ[1]
    pf.Δⱼ = JxX1 + pf.M1x + pf.M1x_a + JxX2
    for i=2:N
        del = Yinv*(pf.buffer_Jx[i]*A[i])*Δ[i]
        @__dot__ pf.Δⱼ += del
    end
    nothing
end

"""
$(TYPEDSIGNATURES)

Experimental implementation of parametric linear multistep methods.
"""
function (pf::PLMsFunctor)(hbuffer, tbuffer, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP, pval)

    copyto!(pf.t, tbuffer)                      # copy time buffer
    update_coefficients!(pf.plms, tbuffer[1])   # update coefficients in linear multistep method
    set_cycle_X!(pf, Xⱼ)                        # update X, rP & P is set in relax
    eval_cycle_Jx!(pf)                          # compute Jx for new time and state (X,P,t) tuple
    eval_cycle_Jp!(pf)                          # compute Jp for new time and state (X,P,t) tuple
    compute_sum_Jp!(pf)                         # set pf.sJp

    @__dot__ pf.x0 = mid(Xⱼ)                    # set x0
    compute_δₖ!(pf)                              # set pf.δₖ

    refine_X!(pf, A, Δⱼ)                        # compute X storing to first position w/o cycling
    @__dot__ pf.x = mid(pf.X)                   # set x
    compute_Δₖ!(Δ, pf)                           # compute Δₖ storing to the first position without cycling

    nothing
end
