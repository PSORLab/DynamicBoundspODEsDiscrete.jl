"""
$(TYPEDEF)

An integrator ....

An elastic array is Y

$(TYPEDFIELDS)
"""
mutable struct DiscretizeRelax{F,X,T,P,Q,K} <: AbstractODERelaxIntegator
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
    "LEPUS gamma constant for Hermite-Obreschkoff"
    γHO::Float64
    "LEPUS gamma constant for parametric linear multistep methods"
    γPILMS::Float64
    "Parametric Linear Multistep Method Storage"
    methodPLMS::PILMS{N,K}
    "Hermite Obreschkoff Storage"
    methodHO::HermiteObreschkoff{P,Q}
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
    "Number of state variables"
    nx::Int
    "Number of decision variables"
    np::Int
    "Relaxation Type"
    type
end
function DiscretizeRelax(d::ODERelaxProb;
                         typeHO::PILMS = PILMS(1)
                         typePILMS::HermiteObreschkoff = HermiteObreschkoff(0,0),
                         repeat_limit = 50, step_limit = 20000,
                         tol = 1E-3, hmin = 1E-13,
                         typeHO =:lohner, relax = false)
    nx = d.nx
    np = d.np
    p = d.p
    pL = d.pL
    pU = d.pU
    f! = d.f!

    # build functors for evaluating Taylor coefficients
    if relax == true
        set_tf! = TaylorFunctor!(f!, nx, np, k, zero(MC{np,NS}), zero(Float64))
        jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(MC{np,NS}), zero(Float64))
    else
        set_tf! = TaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
        jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
    end
    real_tf! = TaylorFunctor!(f!, nx, np, k, zero(Float64), zero(Float64))

    γHO = ### typeHO
    γPILMS = ### typePILMS

    A = QRStack(nx)
    storage = ElasticArray()
    sizehint!(storage, nx+np, 1000)   # suggest 1000*(nx+np) steps likely

    if (tsupports[1] == 0.0)
        # first support is at time index 1
        support_dict = ImmutableDict(d.support_dict, 1 => 1)
    else
        # meaningless placeholder
        support_dict = ImmutableDict(d.support_dict, -1 => -1)
    end
    error_code = RELAXATION_NOT_CALLED

    return DiscretizeRelax{F,X,T}(d.x0, p, pL, pU, d.tspan, d.tsupports, real_tf!,
                                  set_tf!, real_tf!, A, γHO, γPILMS, repeat_limit,
                                  step_limit, tol, hmin, storage, nx, np, type)
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
function single_step!(stf!, rtf!, dtf!, hj_in, A::QRStack, repeat_limit, hmin, nx, Yⱼ,
                      f, γ, Δ, ∂f∂y_in)

    status_flag = COMPLETED
    hj = hj_in
    hj1 = 0.0

    # validate existence & uniqueness
    hⱼ, Ỹⱼ, f̃, step_flag = existence_uniqueness(stf!, Yⱼ, hⱼ, hmin, f, ∂f∂y_in)
    if ~step_flag
        status_flag = NUMERICAL_ERROR
        return status_flag, hj, hj1
    end

    # repeat until desired tolerance is otained or repetition limit is hit
    count = 1
    while (hj > hmin) && (count < repeat_limit)

        yⱼ₊₁, Yⱼ₊₁ = parametric_lohners!(stf!, rtf!, dtf!, hⱼ, Ỹⱼ, Yⱼ, yⱼ, A, Δ)
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

    return status_flag, hj, hj1, yⱼ₊₁, zⱼ₊₁, Yⱼ₊₁
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

set_Δ!(Δ::Vector{Vector{Interval{Float64}}}, out::ElasticArray{Interval{Float64},2})
set_Δ!(Δ::Vector{Vector{MC{N,NS}}}, out::ElasticArray{MC{N,NS},2})

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
        support_indx += 1
    else
        next_support = tsupports[1]
    end

    # initialize QR type storage
    set_Δ!(d.Δ, d.storage)
    reinitialize!(d.A)

    # Begin integration loop
    hlast = 0.0
    hnext = tmax - t
    nsteps = 1
    while sign_tstep*t < sign_tstep*tmax

        # max step size is min of predicted, when next support point occurs,
        # or the last time step in the span
        hnext = min(hnext, next_support - t, tmax - t)

        # perform step size calculation and update bound information
        step_flag, hlast, hnext = single_step!(d.set_tf!, d.real_tf!, d.jac_tf!, hnext,
                                               d.A, d.repeat_limit, d.hmin, d.nx)

        # advance step counters
        t += hlast
        nsteps += 1

        # throw error if limit exceeded
        if nsteps > d.step_limit
            d.error_code = NUMERICAL_ERROR
            break
        elseif step_flag !== COMPLETED
            d.error_code = NUMERICAL_ERROR
            break
        end

        #unpack variables computed in single step
        # TODO
    end

    if d.error_code === EMPTY
        d.error_code = COMPLETED
    end

    nothing
end
