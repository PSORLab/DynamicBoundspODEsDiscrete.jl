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
    β::Vector{Float64}
    "An upper triangular matrix used to compute g part of coefficients"
    c::Matrix{Float64}
    "A vector of length N+1 used to store the gcomponent of the coefficients"
    g::Vector{Float64}
end
function PLMS(x::Val{N}, s::T) where {N, T<:AbstractLinearMethod}
    c = CircularBuffer{Float64}(N)
    fill!(c, 0.0)
    PLMS{N,T}(c, zeros(Float64,N), zeros(Float64,N), zeros(Float64,N,N), zeros(Float64,N))
end

@generated function seed_tuple(u::Val{U}, n::Val{N}) where {U,N}
    expr = Expr(:tuple)
    expr.args = [i == U ? 1.0 : 0.0 for i in 1:N]
    return expr
end

#=
"""
$(TYPEDSIGNATURES)

Computes the cofficients of the N-step Adams Moulton method from the cofficients.
"""
function compute_coefficients!(x::PLMS{N, AdamsMoulton}) where N
    t = x.times
    x.β[1] = 1.0
    for i in 2:N
        x.β[i] = x.β[i]*(t[1] - t[i+1])/(t[2] - t[i+2])
    end
    hn = (t[1] - t[2])
    for q = 1:N
        c[0,q] = 1/q
    end
    for q = 1:N
        for j = 1:N
            if j == 1
                c[1,q] = 1/q
            elseif j >= q
                c[q, j] = c[j-1, q] - c[j-1,q+1]*hn/(t[1] - t[j+1])
            else
                break
            end
        end
        g[j] = c[j,1]
    end
    map!(j -> seed_tuple(Val(j),Val(N)), x.φ[1], 1:N)
    for j=1:k
        x.φ[j+1] = x.φ[j] - x.β[j]
    end
    nothing
end
=#

"""
$(TYPEDEF)

Functor used to evaluate an N-step PLM method.
"""
struct PLMsFunctor{N,T,S,JX,JP}
    "PLMS Storage"
    plms::PLMS{N,T}
    "Circular Buffer for Holding Jx's"
    buffer_Jx::CircularBuffer{Matrix{S}}
    "Circular Buffer for Holding Jp's"
    buffer_Jp::CircularBuffer{Matrix{S}}
    "Circular Buffer for state bounds/relaxations X's"
    X::CircularBuffer{Vector{S}}
    "Circular Buffer for times"
    t::CircularBuffer{Float64}
    "Holds parameter values P"
    P::Vector{S}
    "Holds parameter values shift by reference, P - p"
    rP::Vector{S}
    "Number of state variables"
    nx::Int
    "Number of decision variables"
    np::Int
    "Jacobian of the rhs w.r.t to the state variables"
    Jx!::JX
    "Jacobian of the rhs w.r.t to the descision variables"
    Jp!::JP
    "Temporary Storage for sum of the Jacobian w.r.t p "
    sJp::Matrix{S}
end
function PLMsFunctor(s::S, plms::PLMS{N,T}, Jx!::JX, Jp!::JP,
                     nx::Int, np::Int) where {JX, JP, S, N, T<:AbstractLinearMethod}
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
    PLMsFunctor{N,T,S,JX,JP}(plms, buffer_Jx, buffer_Jp, X,
                             t, P, rP, nx, np, Jx!, Jp!, sJp)
end


function update_coefficients!(pf::PLMsFunctor{N,T}, t0::Float64) where {N, T<:AbstractLinearMethod}
    pushfirst!(pf.plms.times, t0)
    compute_coefficients!(x.plms)
    nothing
end

function set_cycle_X!(pf, Yⱼ)
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

function compute_δₖ!(pf::PILMsFunctor)
end

#=
function refine_X!(pf::PILMsFunctor)
    pf.X = pf.x0 + (Jknx - mid(Jknx))*(X0 - x0) + pf.sJp*pf.rP + pf.δₖ
           + ((I - Jk1nx)*Aₖ₋₁)*Δₖ₋₁ + sum((Jkjnx*Akj)*Δₖⱼ for j=2:n))
end

function compute_Δₖ!(pf::PILMsFunctor)
    Yinv = inv((I - mid(Jknx))*Aₖ)
    Δ = Yinv*()*(X-x) + (Yinv*((I - Jk1nx)*Aₖ₋₁))*Δₖ₋₁ + Yinv*Jkp*pf.rP
        + Yinv*δₖ + sum(Yinv*(Jkjnx*Akj)*Δₖⱼ for j=2:n)
end
=#

"""
$(TYPEDSIGNATURES)

Experimental implementation of parametric linear multistep methods.
"""
function parametric_pilms!(pf::PLMsFunctor, hⱼ, t0, Ỹⱼ, Yⱼ, A::CircularBuffer{QRDenseStorage}, Δ)

    update_coefficients!(pf.plms, hⱼ)     # update coefficients in linear multistep method
    set_cycle_X!(pf, Yⱼ)                  # update X, rP & P is set in relax
    eval_cycle_Jx!(pf)                    # compute Jx for new time and state (X,P,t) tuple
    eval_cycle_Jp!(pf)                    # compute Jp for new time and state (X,P,t) tuple
    compute_sum_Jp!(pf)                   # set pf.sJp
    compute_δₖ!(pf)                       # set pf.δₖ

    @__dot__ pf.x0 = mid(pf.X)            # set x0
    refine_X!(pf)                         # compute X storing to first position w/o cycling
    @__dot__ pf.x = mid(pf.X)             # set x
    compute_Δₖ!(Δ, pf)                    # compute Δₖ storing to the first position without cycling

    nothing
end
