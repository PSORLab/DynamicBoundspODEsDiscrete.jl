abstract type AbstractLinearMethod end
struct AdamsMoulton <: AbstractLinearMethod end
struct BDF <: AbstractLinearMethod end

"""
$(TYPEDEF)

A structure which holds an N-step parametric linear method of

$(TYPEDFIELDS)
"""
mutable struct PLMS{N,T<:AbstractLinearMethod} <: AbstractStateContractorName
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
    t = CircularBuffer{Float64}(N+1)
    fill!(t, 0.0)
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
    PLMS{N,T}(t, zeros(Float64,N), β, zeros(Float64,N,N), zeros(Float64,N), φ)
end
PLMS(x::Int, s::T) where {N, T<:AbstractLinearMethod} = PLMS(Val(x),s)

state_contractor_k(m::PLMS{N,T}) where {N,T<:AbstractLinearMethod} = N+1
state_contractor_γ(m::PLMS) = 1.0
state_contractor_steps(m::PLMS{N,T}) where {N,T<:AbstractLinearMethod} = N

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
        @__dot__ x.β[i] = x.β[i-1]*(t[1] - t[i])/(t[2] - t[i])
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
                x.φ[j][i] = x.φ[j-1][i] .- x.β[j-1][i+1].*x.φ[j-1][i+1]
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
struct PLMsFunctor{F,N,T,S,JX,JP} <: AbstractStateContractor
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
    h::Float64
end
function PLMsFunctor(s::S, plms::PLMS{N,T}, f!::F, Jx!::JX, Jp!::JP,
                     nx::Int, np::Int, h::Float64) where {F, JX, JP, S, N, T<:AbstractLinearMethod}
    buffer_Jx = CircularBuffer{Matrix{S}}(N)
    buffer_Jp = CircularBuffer{Matrix{S}}(N)
    X = CircularBuffer{Vector{S}}(N)
    refx  = CircularBuffer{Vector{Float64}}(N)
    t = CircularBuffer{Float64}(N)
    fill!(buffer_Jx, zeros(S, nx, nx))
    fill!(buffer_Jp, zeros(S, nx, np))
    fill!(X, zeros(S, nx))
    fill!(t, 0.0)
    P = zeros(S,np)
    rP = zeros(S,np)
    p = zeros(Float64,np)
    sJp = zeros(S, nx, np)
    δₖ = zeros(S, nx)
    Z = zeros(Float64, nx)
    x0 = zeros(Float64, nx)
    x = zeros(Float64, nx)
    M1x = zeros(Float64,nx)
    PLMsFunctor{F,N,T,S,JX,JP}(plms, buffer_Jx, buffer_Jp, X, refx,
                             t, P, rP, p, nx, np, f!, Jx!, Jp!, sJp, δₖ,
                             Z, x0, x, M1x, h)
end

function state_contractor(m::PLMS{N,T}, f!::F, Jx!::JX, Jp!::JP, nx::Int, np::Int, style::S, s, h::Float64) where {F,N,T,JX,JP,S}
    println("s: $s, Jx!: $(Jx!), Jp!: $(Jp!)")
    PLMsFunctor(style, m, f!, Jx!, Jp!, nx, np, h)
end


function compute_coefficients!(pf::PLMS{1,T}) where T<:AbstractLinearMethod
    pf.coeffs[1] = 1.0
    nothing
end

function compute_coefficients!(pf::PLMS{2,T}) where T<:AbstractLinearMethod
    pf.coeffs = [0.5; 0.5]
    nothing
end

function compute_coefficients!(pf::PLMS{3,T}) where T<:AbstractLinearMethod
    pf.coeffs = [5.0/12.0; 2.0/3.0; -1.0/12.0]
    nothing
end

function compute_coefficients!(pf::PLMS{4,T}) where T<:AbstractLinearMethod
    pf.coeffs = [9.0/24.0; 19.0/24.0; -5.0/24.0; 1.0/24.0]
    nothing
end

function compute_coefficients!(pf::PLMS{5,T}) where T<:AbstractLinearMethod
    pf.coeffs = [251.0/720.0; 646.0/720.0; -264.0/720.0; 106.0/720.0; -19.0/720.0]
    nothing
end

function update_coefficients!(pf::PLMsFunctor{F,N,T,S,JX,JP}, t0::Float64) where {F,N,T,S,JX,JP}
    if pf.h > 0
        println("ran fixed compute...")
        compute_coefficients!(pf.plms)
    else
        pushfirst!(pf.plms.times, t0)
        compute_coefficients!(pf.plms)
    end
    nothing
end

function set_cycle_X!(pf::PLMsFunctor, Yⱼ)
    println("Yⱼ: $(Yⱼ)")
    println("pf.X: $(pf.X)")
    println("pf.X.buffer: $(pf.X.buffer)")
    pf.X.first = (pf.X.first == 1 ? pf.X.length : pf.X.first - 1)
    copyto!(pf.X.buffer[pf.X.first], 1, Yⱼ, 1, pf.nx)
    nothing
end
function eval_cycle_Jx!(pf::PLMsFunctor)
    println("pf.Jx!: $(pf.Jx!)")
    println("pf.buffer_Jx: $(pf.buffer_Jx)")
    println("first(pf.X): $(first(pf.X))")
    println("pf.P: $(pf.P)")
    println("first(pf.t): $(first(pf.t))")
    eval_cycle!(pf.Jx!, pf.buffer_Jx, first(pf.X), pf.P, first(pf.t))
end
eval_cycle_Jp!(pf::PLMsFunctor) = eval_cycle!(pf.Jp!, pf.buffer_Jp, first(pf.X), pf.P, first(pf.t))

function compute_sum_Jp!(pf::PLMsFunctor)
    map!((x,y) -> x.*y, pf.buffer_Jp, pf.plms.coeffs, pf.buffer_Jp)
    pf.sJp .= pf.buffer_Jp[1]
    for i = 2:length(pf.buffer_Jp)
        pf.sJp .+= pf.buffer_Jp[i]
    end
    nothing
end

function compute_δₖ!(pf::PLMsFunctor{N,T,S,JX,JP}, fk) where {N,T,S,JX,JP}
    println("pf.δₖ: $(pf.δₖ)")
    println("pf.x0: $(pf.x0)")
    println("pf.x: $(pf.x)")
    println("fk: $(fk)")
    println("f.M1x: $(pf.M1x)")
    println("pf.refx: $(pf.refx)")
    println("pf.p: $(pf.p)")
    println("pf.t: $(pf.t)")
    println("pf.plms.coeffs: $(pf.plms.coeffs)")
    @__dot__ pf.δₖ = pf.x0 - pf.x + fk
    for i=1:N
        pf.f!(pf.M1x, pf.refx[i], pf.p, pf.t[i])
        println("pf.M1x: $(pf.M1x)")
        pf.δₖ += pf.plms.coeffs[i]*pf.M1x
    end
    nothing
end

function refine_X!(pf::PLMsFunctor, A, Δⱼ)
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

function compute_Δₖ!(pf::PLMsFunctor, A, Δⱼ)
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
function (pf::PLMsFunctor)(hbuffer, tbuffer, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP, pval, fk)

    println("start of call pf.X: $(pf.X)")
    copyto!(pf.t, tbuffer)                      # copy time buffer
    update_coefficients!(pf, tbuffer[1])        # update coefficients in linear multistep method
    set_cycle_X!(pf, Xⱼ)                        # update X, rP & P is set in relax
    eval_cycle_Jx!(pf)                          # compute Jx for new time and state (X,P,t) tuple
    eval_cycle_Jp!(pf)                          # compute Jp for new time and state (X,P,t) tuple
    compute_sum_Jp!(pf)                         # set pf.sJp

    @__dot__ pf.x0 = mid(Xⱼ)                    # set x0
    pushfirst!(pf.refx, pf.x0)
    compute_δₖ!(pf, fk)                          # set pf.δₖ

    refine_X!(pf, A, Δⱼ)                        # compute X storing to first position w/o cycling
    @__dot__ pf.x = mid(X̃ⱼ)                     # set x
    compute_Δₖ!(Δ, pf)                           # compute Δₖ storing to the first position without cycling

    nothing
end

has_jacobians(d::PLMsFunctor) = false
