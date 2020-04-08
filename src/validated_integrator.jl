"""
$(TYPEDEF)

An integrator ....

An elastic array is Y

$(TYPEDFIELDS)
"""
mutable struct DiscretizeRelax{F,X,T} <: AbstractODERelaxIntegator
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
    "Maximum number of integration steps"
    step_limit::Int
    "Error tolerance"
    tol::Float64
    "Minimum step size"
    hmin::Float64
    "Stores solution Y = (X,P) for each time"
    storage::ElasticArray{T,2}
    "Support index to storage dictory"
    support_dict::ImmutableDict{Int,Int}
    "Holds data for numeric error encountered in integration step"
    error_code::TerminationStatusCode
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

    qr_stack = QRStack(nx)
    storage = ElasticArray()
    sizehint!(storage, nx+np, 1000)                   # suggest 1000*(nx+np) steps likely

    return DiscretizeRelax{F,X,T}()
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
function single_step!(hj_in, A::QRStack, Yⱼ, repeat_limit,
                      f, γ, hmin, nx)

    status_flag = COMPLETED
    hj = hj_in
    hj1 = 0.0

    # validate existence & uniqueness
    hⱼ, Ỹⱼ, f̃, step_flag = existence_uniqueness(tf!, Yⱼ, hⱼ, hmin, f, ∂f∂y_in)
    if ~step_flag
        status_flag = NUMERICAL_ERROR
        return status_flag, hj, hj1
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

    # updates Aj for next step
    advance!(A)

    return status_flag, hj, hj1
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

function compute_y0!(out::ElasticArray{Interval{Float64},2}, x0, p, pL, pU, nx, np)
    @__dot__ P = Interval(pL, pU)
    Xout = x0(P)
    copyto!(view(out, 1:nx, 1), 1, Xout, 1, nx)
    copyto!(view(out, (nx):(nx+np), 1), nx+1, P, 1, np)
    nothing
end

function compute_y0!(out::ElasticArray{MC{N,NS},2}, x0, p, pL, pU, nx, np) where N
    @__dot__ P = MC{N,NS}(p, Interval(pL, pU), 1:np)
    Xout = x0(P)
    copyto!(view(out, 1:nx, 1), 1, Xout, 1, nx)
    copyto!(view(out, (nx):(nx+np), 1), nx+1, P, 1, np)
    nothing
end

function relax!(d::DiscretizeRelax)

    # Functor set P and P - p values for calculations
    set_p!(d.jac_tf!, p, pL, pU)

    # Compute initial condition values
    compute_y0!(d.storage, d.x0f, p, pL, pU, nx, np)

    # Get initial time and integration direction
    t = d.tspan[1]
    tmax = d.tspan[1]
    sign_tstep = copysign(1, tmax-t)

    # Computes maximum step size to take (either hit support or max time)
    support_indx = 1
    if (tsupports[1] == 0.0)
        next_support = tsupports[2]
        d.support_dict = ImmutableDict(d.support_dict, support_indx => 1)
        support_indx += 1
    else
        next_support = tsupports[1]
    end

    # Begin integration loop
    hlast = 0.0
    hnext = tmax - t
    nsteps = 1
    while sign_tstep*t < sign_tstep*tmax

        # max step size is min of predicted, when next support point occurs,
        # or the last time step in the span
        hnext = min(hnext, next_support - t, tmax - t)

        # perform step size calculation and update bound information
        hlast, hnext = single_step!(hnext, d.A, d.repeat_limit)

        # advance step counters
        t += hlast
        nsteps += 1

        # throw error if limit exceeded
        if nsteps > d.step_limit
            d.error_code = NUMERICAL_ERROR
            break
        end

        #unpack variables computed in single step
        # TODO
    end

    nothing
end
