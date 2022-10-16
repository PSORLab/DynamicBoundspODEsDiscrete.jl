export AdamsMoulton

struct AdamsMoulton <: AbstractStateContractorName
    steps::Int
end
mutable struct AdamsMoultonFunctor{S} <: AbstractStateContractor
    f!
    Jx!
    Jp!
    nx
    np
    method_step::Int64
    η::Interval{Float64}
    μX::Vector{S}
    ρP::Vector{S}
    Dk::Vector{S}
    Xold_computed::Vector{S}
    Xk::Vector{Vector{S}}
    fk1::Vector{S}
    fk_val::Vector{Float64}
    Jxsto::FixedCircularBuffer{Matrix{S}}
    Jpsto::FixedCircularBuffer{Matrix{S}}
    Jxsum::Matrix{S}
    Jpsum::Matrix{S}
    Jxvec::Vector{S}
    Jpvec::Vector{S}
    Ainv_Jxvec::Vector{S}
    Ainv_Jpvec::Vector{S}
    Ainv_fk_val::Vector{S}
    Ainv_Xk2::Vector{S}
    Ainv_Rk::Vector{S}
    fval::FixedCircularBuffer{Vector{Float64}}
    fk_apriori::FixedCircularBuffer{Vector{S}}
    Rk::Vector{S}
    Y0::Matrix{Float64}
    Y::Matrix{Float64}
    Jxmid_sto::Matrix{S}
    precond::LinearAlgebra.LU{Float64,Array{Float64,2}}
    JxAff::Matrix{S}
    YJxAff::Matrix{S}
    YJxΔx::Vector{S}
    Xj_delta::Vector{S}
    X_last::Vector{S}
    Ysumx::Vector{S}
    YsumP::Matrix{S}
    YJpΔp::Vector{S}
    coeffs::Vector{Float64}
    Δⱼ₊₁::Vector{S}
    is_adaptive::Bool
    γ::Float64
    lohners_start::LohnersFunctor
    A_Q::FixedCircularBuffer{Matrix{Float64}}
    A_inv::FixedCircularBuffer{Matrix{Float64}}
    Δ::FixedCircularBuffer{Vector{S}}
    X::FixedCircularBuffer{Vector{S}}
    xval::FixedCircularBuffer{Vector{Float64}}
    Δx::Vector{S}
    constant_state_bounds::Union{Nothing, ConstantStateBounds}
end
function set_constant_state_bounds!(d::AdamsMoultonFunctor, v)
    d.constant_state_bounds = v
    set_constant_state_bounds!(d.lohners_start, v)
    nothing
end

function AdamsMoultonFunctor(f::F, Jx!::JX, Jp!::JP, nx::Int, np::Int, s::S, t::T, steps::Int, lohners_start) where {F,JX,JP,S,T}
    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)

    method_step = steps
    Dk = zeros(S, nx)

    lu_mat = zeros(nx, nx)
    for i = 1:nx
        lu_mat[i,i] = 1.0
    end
    precond = lu(lu_mat)
    Y0 = zeros(nx, nx)
    Y = zeros(nx, nx)
    Jxmid_sto = zeros(S, nx, nx)
    YJxAff = zeros(S, nx, nx)
    JxAff = zeros(S, nx, nx)
    Xj_delta = zeros(S, nx)
    X_last = zeros(S, nx)
    YJxΔx = zeros(S, nx)
    Ysumx = zeros(S, nx)
    YsumP = zeros(S, nx, nx)
    YJpΔp = zeros(S, nx)

    Jxsto = FixedCircularBuffer{Matrix{S}}(method_step)
    Jpsto = FixedCircularBuffer{Matrix{S}}(method_step)
    for i = 1:method_step
        push!(Jxsto, zeros(S, nx, nx))
        push!(Jpsto, zeros(S, nx, np))
    end
    Jxsum = zeros(S, nx, nx)
    Jpsum = zeros(S, nx, np)
    Jxvec = zeros(S, nx)
    Jpvec = zeros(S, nx)
    Ainv_Jxvec = zeros(S, nx)
    Ainv_Jpvec = zeros(S, nx)
    Ainv_fk_val = zeros(S, nx)
    Ainv_Xk2 = zeros(S, nx)
    Ainv_Rk = zeros(S, nx)

    Xold_computed = zeros(S, nx)
    Δx = zeros(S, nx)
    fk1 = zeros(S, nx)
    fk_val = zeros(Float64, nx)
    Xk = Vector{S}[zeros(S, nx)]

    fval = FixedCircularBuffer{Vector{Float64}}(method_step)
    fk_apriori = FixedCircularBuffer{Vector{S}}(method_step)
    A_Q = FixedCircularBuffer{Matrix{Float64}}(method_step)
    A_inv = FixedCircularBuffer{Matrix{Float64}}(method_step)
    Δ = FixedCircularBuffer{Vector{S}}(method_step)
    X = FixedCircularBuffer{Vector{S}}(method_step)
    xval = FixedCircularBuffer{Vector{Float64}}(method_step)
    #@show method_step
    for i = 1:method_step
        push!(xval, zeros(Float64, nx))
        push!(fval, zeros(Float64, nx))
        push!(fk_apriori, zeros(S, nx))
        push!(A_Q, Float64.(Matrix(I, nx, nx)))
        push!(A_inv, Float64.(Matrix(I, nx, nx)))
        push!(X, zeros(S, nx))
        push!(Δ, zeros(S, nx))
        push!(Xk, zeros(S, nx))
    end

    Rk = zeros(S, nx)
    coeffs = zeros(Float64, method_step + 1)
    Δⱼ₊₁ = zeros(S, nx)

    is_adaptive = false
    γ = 0.0

    AdamsMoultonFunctor{S}(f, Jx!, Jp!, nx, np, method_step, η, μX, ρP,
                           Dk, Xold_computed, Xk, fk1, fk_val,
                           Jxsto, Jpsto, Jxsum, Jpsum, Jxvec, Jpvec, Ainv_Jxvec, Ainv_Jpvec, Ainv_fk_val,
                           Ainv_Xk2, Ainv_Rk, fval, fk_apriori, Rk, Y0, Y, Jxmid_sto, precond, JxAff, YJxAff,
                           YJxΔx, Xj_delta, X_last, Ysumx, YsumP, YJpΔp, coeffs, Δⱼ₊₁,
                           is_adaptive, γ, lohners_start, A_Q, A_inv,
                           Δ, X, xval, Δx, nothing)
end

function compute_coefficients!(c::ContractorStorage{S}, d::AdamsMoultonFunctor{S}, h::Float64, t::Float64, s::Int) where S
    @unpack is_adaptive, coeffs = d

    if !is_adaptive
        if s == 1
            coeffs[1] = 1.0
            coeffs[2] = 0.5
        elseif s == 2
            coeffs[1] = 0.5
            coeffs[2] = 0.5
            coeffs[3] = -1.0/12.0
        elseif s == 3
            coeffs[1] = 5.0/12.0
            coeffs[2] = 8.0/12.0
            coeffs[3] = -1.0/12.0
            coeffs[4] = -1.0/24.0
        elseif s == 4
            coeffs[1] = 9.0/24.0
            coeffs[2] = 19.0/24.0
            coeffs[3] = -5.0/24.0
            coeffs[4] = 1.0/24.0
            coeffs[5] = -19.0/720.0
        else
           error("order greater than 4 for fixed size PILMS currently not supported")
        end
    else
        compute_adaptive_coeffs!(d, h, t, s)
    end
    d.γ = coeffs[s + 1]
    c.γ = d.γ

    nothing
end

function compute_Rk!(d::AdamsMoultonFunctor{T}, c::ContractorStorage{T}, h, s) where T<:Number
    @unpack fk_apriori, Rk, method_step, γ = d

    cycle_copyto!(fk_apriori, c.fk_apriori, c.step_count - 1)
    @. Rk = fk_apriori[1]
    for i = 2:s
        @. Rk = Rk ∪ fk_apriori[i]
    end
    @. Rk *= γ*h^(method_step+1)

    nothing
end

# TODO: Check block
function compute_real_sum!(d::AdamsMoultonFunctor{T}, c::ContractorStorage{T}, r::StepResult{T}, h::Float64, t::Float64, s::Int) where T<:Number

    @unpack Dk, coeffs, fval, f!, X_last = d
    @unpack Xj_0, pval, X_computed = c
    #println("real sum Xj_0 = $(Xj_0)")

    #=
    @. X_last = X_computed
    @show mid.(X_last), pval, t, h

    cycle_eval!(f!, fval, mid.(Xj_0), pval, t)
    @. Dk = mid(X_last) # TODO: REPLACE WITH (X_K-1) NOT EXISTENCE TEST... Maybe fixed...
    for i = 1:s
        @show coeffs[i]
        @. Dk += h*coeffs[i]*fval[i]
    end
    =#
    
    nothing
end

# Compute components of sum for prior timesteps --> then update original
function compute_jacobian_sum!(d::AdamsMoultonFunctor{T}, c::ContractorStorage{T}, h::Float64, t::Float64, s::Int) where T<:Number

    @unpack Xj_delta, Jx!, Jp!, JxAff, Jxsto, Jpsto, Jxsum, Jpsum, Jpvec, μX, ρP, η, coeffs = d
    @unpack Xj_0, Xj_apriori, xval, P, pval, A_Q, Δ = c

    μ!(μX, Xj_0, xval, η)
    ρ!(ρP, P, pval, η)
    cycle_eval!(Jx!, Jxsto, μX, ρP, t)
    cycle_eval!(Jp!, Jpsto, μX, ρP, t)

    @. JxAff = Jxsto[1]
    @. Xj_delta = Xj_apriori - mid(Xj_apriori)

    #=
    @show size(Jxsum)
    @show size(Jxsto[1])
    @show size(Xj_delta)
    Jxsum = h*coeffs[1]*(Jxsto[1]*Xj_delta)
    if s > 1
        println("s = $s")
        @show size(((I + h*coeffs[2]*Jxsto[2])*A_Q[1])*Δ[1])
        @show size(Jxsum)
        Jxsum += ((I + h*coeffs[2]*Jxsto[2])*A_Q[1])*Δ[1]
    end
    for i = 3:s
        println("s = $s of 3")
        Jxsum += h*coeffs[i]*(Jxsto[i]*A_Q[i])*Δ[i]
    end
    =#

    nothing
end

@inline function union_mc_dbb(x::MC{N,T}, y::MC{N,T}) where {N, T <: RelaxTag}
    cv_MC = min(x, y)
    cc_MC = max(x, y)
    return MC{N, NS}(cv_MC.cv, cc_MC.cc, Interval(cv_MC.Intv.lo, cc_MC.Intv.hi),
                     cv_MC.cv_grad, cc_MC.cc_grad, x.cnst && y.cnst)
end
union_mc_dbb(x::Interval{T}, y::Interval{T}) where T = x ∪ y


function compute_X!(d::AdamsMoultonFunctor{T}, c::ContractorStorage{S}) where {S, T<:Number}
    @unpack Jx!, Jp!, Jxsum, Jpsum, Jxvec, Jpvec, Dk, Rk, f!, X_last, μX, ρP, X, method_step, Δx, coeffs = d 
    @unpack Xold_computed, Xk, fk1, fk_val, fk_apriori, γ, Jxmid_sto, Jxsto, Δ, A_inv = d
    @unpack Ainv_fk_val, Ainv_Jpvec, Ainv_Jxvec, Ainv_Xk2, Ainv_Rk, Δⱼ₊₁ = d
    @unpack X_computed, xval_computed, Xj_apriori, rP, Xj_0, xval, pval, hj, A_Q, A_inv, B = c
    t = c.times[1]

    s = min(c.step_count - 1, method_step)
    cycle_copyto!(fk_apriori, c.fk_apriori, c.step_count - 1) # TODO: Fix when variable stepsize
    @. Rk = fk_apriori[1]
    for i = 2:s
        @. Rk = union_mc_dbb(Rk, fk_apriori[i])
    end
    @. Rk *= γ*hj^(method_step+1)


    @. Xk[1] = Xj_apriori
    @. Xk[2] = Xj_0
    for j = 3:method_step
        @. Xk[j] = X[j - 2]
    end

    @. Xold_computed = Xj_apriori
    @. X_computed = Xj_apriori

    for i = 1:2
        @. Xk[1] = X_computed
        f!(fk_val, xval, pval, t)
        Jx!(Jxsum, Xk[1], ρP, t)
        Jp!(Jpsum, Xk[1], ρP, t)

        @. Δx = Xk[1] - xval
        Aj_inv = A_inv[2] # TODO: Not sure if set correct...
        Δj = Δ[2]         # TODO: Not sure if set correct...
        mul!(Δx, Aj_inv, Δj)
        mul!(Jxvec, Jxsum, Δx)

        @. Jpsum = hj*coeffs[1]*Jpsum
        #for i = 2:s
        #    @. Jpsum += h*coeffs[i]*Jpsto[i]
        #end
        mul!(Jpvec, Jpsum, ρP)
 
        @. Xold_computed = X_computed
        @. X_computed = Xk[2] + Rk + hj*coeffs[1]*(fk_val + Jpvec + Jxvec)
        for j = 1:method_step
            f!(fk1, Xk[j + 1], ρP, t)
            @. X_computed += hj*coeffs[j + 1]*fk1
        end 

        # Finish computing X
        @. X_computed = X_computed ∩ Xold_computed
        contract_constant_state!(X_computed, d.constant_state_bounds)

        # Compute x
        @. xval_computed = mid(X_computed)

        # Compute Delta
        if s > 1
            Ak_m_1 =  A_Q[2]
            mul!(Jxmid_sto, Jxsto[1], Ak_m_1)
            @. B = mid(Jxmid_sto)
            calculateQ!(A_Q[1], B)
            calculateQinv!(A_inv[1], A_Q[1])
        end
        mul!(Ainv_fk_val, A_inv[1], fk_val)
        mul!(Ainv_Jpvec,  A_inv[1], Jpvec)
        mul!(Ainv_Jxvec,  A_inv[1], Jxvec)
        mul!(Ainv_Xk2, A_inv[1], Xk[2])
        mul!(Ainv_Rk,  A_inv[1], Rk)

        @. Δⱼ₊₁ = hj*coeffs[1]*(Ainv_fk_val + Ainv_Jpvec + Ainv_Jxvec)
        for j = 1:method_step
            f!(fk1, Xk[j + 1], ρP, t)
            mul!(Ainv_fk_val, A_inv[1], fk1)
            @. Δⱼ₊₁ += hj*coeffs[j + 1]*Ainv_fk_val
        end 

    end

    nothing
end

function store_starting_buffer!(d::AdamsMoultonFunctor{T}, c::ContractorStorage{T}, r::StepResult{T}, t::Float64) where T
    @unpack η, X, xval, fk_apriori, Δ, A_Q, A_inv, μX, ρP, Jx!, Jp!, f!, Jxsto, Jpsto, fval = d
    @unpack pval, P, step_count = c

    k = step_count - 1
    cycle_copyto!(X, r.Xⱼ, k)
    cycle_copyto!(xval, r.xⱼ, k)
    cycle_copyto!(fk_apriori, c.fk_apriori, k)
    cycle_copyto!(Δ, r.Δ[1], k)
    cycle_copyto!(A_Q, r.A_Q[1], k)
    cycle_copyto!(A_inv, r.A_inv[1], k)

    # update Jacobian storage
    μ!(μX, r.Xⱼ, r.xⱼ, η)
    ρ!(ρP, P, pval, η)

    cycle_eval!(Jx!, Jxsto, μX, ρP, t)
    cycle_eval!(Jp!, Jpsto, μX, ρP, t)
    cycle_eval!(f!, fval, r.xⱼ, pval, t)

    nothing
end

function (d::AdamsMoultonFunctor{T})(c::ContractorStorage{S}, r::StepResult{S}, count::Int, k) where {S, T<:Number}
    t = c.times[1]
    s = min(c.step_count-1, d.method_step)
    if s < d.method_step
        d.lohners_start(c, r, count, k)
        iszero(count) && store_starting_buffer!(d, c, r, t)
        return nothing
    end

    h = c.hj
    compute_coefficients!(c, d, h, t, s)
    compute_real_sum!(d, c, r, h, t, s)
    compute_jacobian_sum!(d, c, h, t, s)
    compute_X!(d, c)
    return nothing
end

function get_Δ(d::AdamsMoultonFunctor)
    if true
        return copy(get_Δ(d.lohners_start))
    end
    return copy(d.Δⱼ₊₁)
end

function state_contractor(m::AdamsMoulton, f, Jx!, Jp!, nx, np, style, s, h)
    lohners_functor = state_contractor(LohnerContractor{m.steps}(), f, Jx!, Jp!, nx, np, style, s, h)
    AdamsMoultonFunctor(f, Jx!, Jp!, nx, np, style, s, m.steps, lohners_functor)
end
state_contractor_k(m::AdamsMoulton) = m.steps + 1
state_contractor_γ(m::AdamsMoulton) = 1.0  #  Same value as Lohners for initial step then set later...
state_contractor_steps(m::AdamsMoulton) = m.steps
state_contractor_integrator(m::AdamsMoulton) = CVODE_Adams()

function set_γ!(sc::AdamsMoulton, c, ex, result, params)
    nothing
end