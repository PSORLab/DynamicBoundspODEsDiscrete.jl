struct StepParams
    "Error tolerance"
    tol::Float64
    "Minimum step size"
    hmin::Float64
    "Number of decision variables"
    nx::Int
    "LEPUS repetition limit"
    repeat_limit::Int
    "Gamma constant"
    γ::Float64
    "Order of Taylor series approx"
    k::Int
    "Should the second stage be skipped?"
    skip_step2::Bool
end

"""
$(TYPEDEF)

An integrator ....

An elastic array is Y

$(TYPEDFIELDS)
"""
mutable struct DiscretizeRelax{M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY} <: AbstractODERelaxIntegator

    # Problem description
    "Initial Conditiion for pODEs"
    x0f::X
    "Parameter value for pODEs"
    p::Vector{Float64}
    "Lower Parameter Bounds for pODEs"
    pL::Vector{Float64}
    "Upper Parameter Bounds for pODEs"
    pU::Vector{Float64}
    "Number of state variables"
    nx::Int
    "Number of decision variables"
    np::Int
    "Time span to integrate over"
    tspan::Tuple{Float64, Float64}
    "Individual time points to evaluate"
    tsupports::Vector{Float64}

    # Options and internal storage
    "Maximum number of integration steps"
    step_limit::Int
    "Steps taken"
    step_count::Int
    "Stores solution X (from step 2) for each time"
    storage::Vector{Vector{T}}
    "Stores solution X (from step 1) for each time"
    storage_apriori::Vector{Vector{T}}
    "Stores each time t"
    time::Vector{Float64}
    "Support index to storage dictory"
    support_dict::Dict{Int,Int}
    "Holds data for numeric error encountered in integration step"
    error_code::TerminationStatusCode
    "Storage for QR Factorizations"
    A::CircularBuffer{QRDenseStorage}
    "Storage for Δ"
    Δ::CircularBuffer{Vector{T}}
    "Storage for bounds/relaxation of P"
    P::Vector{T}
    "Storage for bounds/relaxation of P - p"
    rP::Vector{T}
    "Relaxation Type"
    style::T
    "Flag indicating that only apriori bounds should be computed"
    skip_step2::Bool

    # Main functions used in routines
    "Functor for evaluating Taylor coefficients over a set"
    set_tf!::TaylorFunctor!{F,K,S,T}
    method_f!::M

    step_result::StepResult{T}
    step_params::StepParams

    new_decision_pnt::Bool
    new_decision_box::Bool
end
function DiscretizeRelax(d::ODERelaxProb, m::SCN; repeat_limit = 50, step_limit = 1000,
                         tol = 1E-5, hmin = 1E-13, relax = false, h = 0.0, skip_step2 = false) where SCN <: AbstractStateContractorName

    γ = state_contractor_γ(m)::Float64
    k = state_contractor_k(m)::Int
    method_steps = state_contractor_steps(m)::Int

    tsupports = d.tsupports
    if ~isempty(tsupports)
        if (tsupports[1] == 0.0)
            support_dict = Dict{Int,Int}(d.support_dict, 1 => 1)
        end
    else
        support_dict = Dict{Int,Int}()
    end
    error_code = RELAXATION_NOT_CALLED

    T = relax ? MC{d.np,NS} : Interval{Float64}
    style = zero(T)
    time = zeros(Float64,1000)
    storage = Vector{T}[]
    storage_apriori = Vector{T}[]
    for i=1:1000
        push!(storage, zeros(T, d.nx))
        push!(storage_apriori, zeros(T, d.nx))
    end
    P = zeros(T, d.np)
    rP = zeros(T, d.np)

    A = qr_stack(d.nx, method_steps)
    Δ = CircularBuffer{Vector{T}}(method_steps)
    fill!(Δ, zeros(T, d.nx))

    set_tf! = TaylorFunctor!(d.f, d.nx, d.np, Val(k), style, zero(Float64))
    state_method = state_contractor(m, d.f, d.nx, d.np, style, zero(Float64))
    #method_f! = LohnersFunctor(d.f, d.nx, d.np, Val(k), style, zero(Float64))

    step_result = StepResult(style, d.nx, d.np, k, h)
    step_params = StepParams(tol, hmin, d.nx, repeat_limit, γ, k, skip_step2)

    return DiscretizeRelax{typeof(state_method), T, Float64, typeof(d.f), k+1,
                           typeof(d.x0), d.nx+d.np}(d.x0, d.p, d.pL, d.pU, d.nx,
                           d.np, d.tspan, d.tsupports, step_limit, 0, storage,
                           storage_apriori, time, support_dict, error_code, A,
                           Δ, P, rP, skip_step2, style, set_tf!, state_method,
                           step_result, step_params, true, true)
end
function DiscretizeRelax(d::ODERelaxProb; kwargs...)
    DiscretizeRelax(d, LohnerContractor{4}(); kwargs...)
end


"""
$(TYPEDSIGNATURES)

Estimates the local excess from as the infinity-norm of the diam of the
kth Taylor cofficient of the prior step.
"""
function estimate_excess(hj, k, fk, γ, nx)
    errⱼ = 0.0
    dₜ = 0.0
    for i=1:nx
        @inbounds dₜ = diam(fk[i])
        errⱼ = (dₜ > errⱼ) ? dₜ : errⱼ
    end
    return γ*(hj^k)*errⱼ
end

function lepus_step_size!(out::StepResult{S}, params::StepParams, k::Int, nx::Int) where {S <: Real}
    out.errⱼ = estimate_excess(out.hj, k, out.f[k], params.γ, nx)
    if out.errⱼ <= out.hj*params.tol
        out.predicted_hj = 0.9*out.hj*(0.5*out.hj/out.errⱼ)^(1/(k-1))
    else
        out.predicted_hj = out.hj*(out.hj*params.tol/out.errⱼ)^(1/(k-1))
        @__dot__ out.f[k] = out.f[k]*(out.predicted_hj/out.hj)^k
        out.hj = out.predicted_hj
        return false
    end
    true
end

"""
$(TYPEDSIGNATURES)

Performs a single-step of the validated integrator. Input stepsize is out.step.
"""
function single_step!(out::StepResult{S}, params::StepParams, sc::M,
                      stf!::TaylorFunctor!, Δ::CircularBuffer{Vector{S}},
                      A::CircularBuffer{QRDenseStorage}, P, rP, p, t) where {M<:AbstractStateContractor, S<:Real}

    k = params.k

    # validate existence & uniqueness
    if ~out.jacobians_set
        set_JxJp!(lf.jac_tf!, out.Xⱼ, P, t)
    end
    existence_uniqueness!(out, stf!, params.hmin, P, t)
    out.Xapriori .= out.unique_result.X
    out.hj = out.unique_result.step
    if ~out.unique_result.confirmed
        out.status_flag = NUMERICAL_ERROR
        return nothing
    end

    # repeat until desired tolerance is otained or repetition limit is hit
    if ~params.skip_step2
        count = 1
        while (out.hj > params.hmin) && (count < params.repeat_limit)

            # perform corrector step
            out.status_flag = sc(out.hj, out.unique_result.X, out.Xⱼ, out.xⱼ, A, Δ, P, rP, p, t)

            # Perform Lepus error control scheme if step size not set
            if out.h <= 0.0
                lepus_step_size!(out, params, k, params.nx) && break
            else
                out.hj = out.h
                out.predicted_hj = out.h
                break
            end
            count += 1
        end

        # updates shifts Aj+1 -> Aj and so on
        pushfirst!(A, last(A))
        pushfirst!(Δ, get_Δ(sc))

        set_x!(out.xⱼ, sc)::Nothing
        set_X!(out.Xⱼ, sc)::Nothing
    else
        out.hj = out.h
        out.xⱼ .= mid.(out.Xapriori)
        out.Xⱼ .= out.Xapriori
    end

    nothing
end

function set_P!(d::DiscretizeRelax{M,Interval{Float64},S,F,K,X,NY}) where {M<:AbstractStateContractor, S, F, K, X, NY}
    @__dot__ d.P = Interval(d.pL, d.pU)
    @__dot__ d.rP = d.P - d.p
    nothing
end

function set_P!(d::DiscretizeRelax{M,MC{N,T},S,F,K,X,NY}) where {M<:AbstractStateContractor, T<:RelaxTag, S <: Real, F, K, X, N, NY}
    @__dot__ d.P = MC{N,NS}.(d.p, Interval(d.pL, d.pU), 1:d.np)
    @__dot__ d.rP = d.P - d.p
    nothing
end

function compute_X0!(d::DiscretizeRelax) #where {X, T <: Number}
    d.storage[1] .= d.x0f(d.P)
    d.storage_apriori[1] .= d.storage[1]
    d.step_result.Xⱼ .= d.storage[1]
    d.step_result.xⱼ .= mid.(d.step_result.Xⱼ)
    nothing
end

function set_Δ!(Δ::CircularBuffer{Vector{T}}, out::Vector{Vector{T}}) where T
    Δ[1] .= out[1] .- mid.(out[1])
    for i in 2:length(Δ)
        fill!(Δ[i], zero(T))
    end
    nothing
end

function DBB.relax!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY}

    set_P!(d) ::Nothing         # Functor set P and P - p values for calculations
    compute_X0!(d)::Nothing     # Compute initial condition values

    # Get initial time and integration direction
    t = d.tspan[1]
    tmax = d.tspan[2]
    sign_tstep = copysign(1, tmax-t)
    d.time[1] = t

    # Computes maximum step size to take (either hit support or max time)
    support_indx = 1
    next_support = Inf
    tsupports = d.tsupports
    if ~isempty(tsupports)
        if (tsupports[1] == 0.0)
            next_support = tsupports[2]
            support_indx += 1
        else
            next_support = tsupports[1]
        end
    end

    # initialize QR type storage
    set_Δ!(d.Δ, d.storage)::Nothing
    reinitialize!(d.A)::Nothing

    # Begin integration loop
    hlast = 0.0
    d.step_result.hj = d.step_result.h > 0.0 ? d.step_result.h : (tmax - t)
    d.step_count = 0

    while sign_tstep*t < sign_tstep*tmax

        # max step size is min of predicted, when next support point occurs,
        # or the last time step in the span
        println("sr.hj = $(d.step_result.hj), ns - t = $(next_support - t), tm - t = $(tmax - t)")
        d.step_result.hj = min(d.step_result.hj, next_support - t, tmax - t)

        # perform step size calculation and update bound information
        single_step!(d.step_result, d.step_params, d.method_f!, d.set_tf!, d.Δ, d.A, d.P, d.rP, d.p, t)::Nothing

        # advance step counters
        t += d.step_result.hj

        # throw error if limit exceeded
        if d.step_count > d.step_limit
            d.error_code = LIMIT_EXCEEDED
            break
        elseif (d.step_result.status_flag !== RELAXATION_NOT_CALLED)
            d.error_code = d.step_result.status_flag
            break
        end
        d.step_count += 1

        # unpack storage
        if d.step_count > length(d.time)-1
            push!(d.storage, copy(d.step_result.Xⱼ))
            push!(d.storage_apriori, copy(d.step_result.Xapriori))
            push!(d.time, t)
        end
        copy!(d.storage[d.step_count+1], d.step_result.Xⱼ)
        copy!(d.storage_apriori[d.step_count+1], d.step_result.Xapriori)
        d.time[d.step_count+1] = t
    end

    # cut out any unnecessary array elements
    resize!(d.storage, d.step_count+1)
    resize!(d.storage_apriori, d.step_count+1)
    resize!(d.time, d.step_count+1)

    if d.error_code === RELAXATION_NOT_CALLED
        d.error_code = COMPLETED
    end

    nothing
end
