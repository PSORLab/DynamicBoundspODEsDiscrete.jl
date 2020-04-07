"""
$(TYPEDEF)

An integrator ....

$(TYPEDFIELDS)
"""
mutable struct DiscretizeRelax{F,X} <: AbstractODERelaxIntegator
    "Right-hand side function for pODEs"
    f::F
    "Initial Conditiion for pODEs"
    x0f::X
    "Parameter value for pODEs"
    p::Vector{Float64}
    "Lower Parameter Bounds for pODEs"
    pL::Vector{Float64}
    "Upper Parameter Bounds for pODEs"
    pU::Vector{Float64}
    "Time span to integrate over"
    tspan::Tuple{Float64, Float64}
    "Individual time points to evaluate"
    tsupports::Vector{Float64}
    "Functor for evaluating Taylor coefficients at real values"
    real_tf!::TaylorFunctor!
    "Functor for evaluating Taylor coefficients over a set"
    set_tf!::TaylorFunctor!
    "Functor for evaluating Jacobians of Taylor coefficients over a set"
    jac_tf!::JacTaylorFunctor!
    "Storage for QR Factorizations"
    A::QRStack
    "LEPUS gamma constant"
    γ::Float64
    "LEPUS repetition limit"
    repeat_limit::Int
    "Error tolerance"
    tol::Float64
    "Minimum step size"
    hmin::Float64
    "Relaxation Type"
    type
end
function DiscretizeRelax(d::ODERelaxProb, relax = false)
    np = d.np
    p = d.p
    pL = d.pL
    pU = d.pU

    # build functors for evaluating Taylor coefficients
    if relax == true
        set_tf! = TaylorFunctor!(f!, nx, np, k, zero(MC{np,NS}), zero(Float64))
        jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(MC{np,NS}), zero(Float64))
    else
        set_tf! = TaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
        jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
    end
    real_tf! = TaylorFunctor!(f!, nx, np, k, zero(Float64), zero(Float64))

    Aⱼ = QRDenseStorage(nx)
    Aⱼ₊₁ = QRDenseStorage(nx)

    return DiscretizeRelax{F,X}
end


"""
$(TYPEDSIGNATURES)

Estimates the local excess from as the infinity-norm of the diam of the
kth Taylor cofficient of the prior step.
"""
function estimate_excess(hj, k, fk, γ, nx)
    errⱼ = 0.0
    dₜ = 0.0
    for i in 1:nx
        @inbounds dₜ = diam(fk[i])
        errⱼ = (dₜ > errⱼ) ? dₜ : errⱼ
    end
    return excess_flag
end



"""
$(TYPEDSIGNATURES)

Performs a single-step of the validated integrator.
"""
function single_step!(hj_in, Yⱼ, Aⱼ₊₁, Aⱼ, next_step, repeat_limit,
                      f, γ, hmin, nx)

    hj = hj_in
    hj1 = 0.0
    # validate existence & uniqueness
    hⱼ, Ỹⱼ, f̃, step_flag = existence_uniqueness(tf!, Yⱼ, hⱼ, hmin, f, ∂f∂y_in)
    if ~success_flag
        return success_flag
    end

    # repeat until desired tolerance is otained or repetition limit is hit
    count = 1
    while (hj > hmin) && (count < repeat_limit)
        yⱼ₊₁, Yⱼ₊₁ = parametric_lohners!(stf!, rtf!, dtf!, hⱼ, Ỹⱼ, Yⱼ, yⱼ, Aⱼ₊₁, Aⱼ, Δⱼ)
        zⱼ₊₁ .= real_tf!.f̃
        Yⱼ₊₁ = jac_tf!.Yⱼ₊₁
        # Lepus error control scheme
        errⱼ = estimate_excess(hj, k, f̃[k], γ, nx)
        if errⱼ <= hj*tol
            hj1 = 0.9*hj*(0.5*hj*tol/errⱼ)^(1/(k-1))
            break
        else
            hj1 = hj*(hj*tol/errⱼ)^(1/(k-1))
            zjp1_temp = zjp1*(hj1/hj)^k
            hj = hj1
            zjp1 .= zjp1_temp
        end
        count += 1
    end

    hj = min(hj, next_step)     # us predict step or support whichever is closest
    copyto!(Aⱼ, Aⱼ₊₁)           # updates Aj for next step
    return success_flag
end

function set_p!(jac_tf!::JacTaylorFunctor!{F,T,MC{N,NS},D}, p, pL, pU) where {F <: Function, T <: Real, N, D}
    for i in 1:length(p)
        jac_tf!.rP[i] = MC{N,NS}.(p[i], Interval(pL[i],pU[i]), i) - p
        nothing
    end
end

function set_p!(jac_tf!::JacTaylorFunctor!{F,T,Interval{Float64},D}, p, pL, pU) where {F <: Function, T <: Real, D}
    @__dot__ jac_tf!.rP = Interval(pL, pU) - p
    nothing
end

function compute_y0!(out::Union{Vector{Interval{Float64}}, SubArray{Interval{Float64},1}},
                     x0, p, pL, pU, nx, np)
    @__dot__ P = Interval(pL, pU)
    xout = x0(P)
    copyto!(out, 1, xout, 1, nx)
    copyto!(out, nx+1, P, 1, np)
    nothing
end

function compute_y0!(out::Union{Vector{MC{N,NS}}, SubArray{MC{N,NS},1}}, x0, p, pL, pU, nx, np) where N
    @__dot__ P = MC{N,NS}(p, Interval(pL, pU), 1:np)
    xout = x0(P)
    copyto!(out, 1, xout, 1, nx)
    copyto!(out, nx+1, P, 1, np)
    nothing
end

function relax!(d::DiscretizeRelax)

    # Functor set P and P - p values for calculations
    set_p!(d.jac_tf!, p, pL, pU)

    # setup storage

    # Compute initial condition values
    compute_y0!(out, d.x0f, p, pL, pU, nx, np)

    integrate_flag &= single_step!()
    step_size_error(integrate_flag)
    nothing
end
