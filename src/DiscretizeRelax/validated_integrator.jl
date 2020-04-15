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
end

"""
$(TYPEDEF)

An integrator ....

An elastic array is Y

$(TYPEDFIELDS)
"""
mutable struct DiscretizeRelax{X,T} <: AbstractODERelaxIntegator

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
    "Stores solution X for each time"
    storage::ElasticArray{T,2}
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

    # Main functions used in routines
    "Functor for evaluating Taylor coefficients over a set"
    set_tf!::TaylorFunctor!
    method_f!::LohnersFunctor

    step_result::StepResult{T}
    step_params::StepParams

    new_decision_pnt::Bool
    new_decision_box::Bool

end
function DiscretizeRelax(d::ODERelaxProb; repeat_limit = 50, step_limit = 1000,
                         tol = 1E-5, hmin = 1E-13, relax = false, k = 4,
                         method_steps = 2, γ = 1.0, h = 0.0)

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
    storage = ElasticArray(zeros(T, d.nx, 1))
    time = zeros(Float64,1)
    P = zeros(T, d.np)
    rP = zeros(T, d.np)

    sizehint!(storage, d.nx, 100000)
    sizehint!(time, 100000)
    A = qr_stack(d.nx, method_steps)
    Δ = CircularBuffer{Vector{T}}(method_steps)
    fill!(Δ, zeros(T, d.nx))

    set_tf! = TaylorFunctor!(d.f, d.nx, d.np, k, style, zero(Float64))
    method_f! = LohnersFunctor(d.f, d.nx, d.np, k, style, zero(Float64))

    step_result = StepResult(style, d.nx, d.np, k, h)
    step_params = StepParams(tol, hmin, d.nx, repeat_limit, γ, k)

    return DiscretizeRelax{typeof(d.x0),T}(d.x0, d.p, d.pL, d.pU, d.nx, d.np, d.tspan,
                                           d.tsupports, step_limit, 0, storage, time,
                                           support_dict, error_code, A, Δ, P, rP, style, set_tf!,
                                           method_f!, step_result, step_params, true, true)
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
    return γ*(hj^k)*errⱼ
end


function lepus_step_size!(out::StepResult{S}, params::StepParams,
                          k::Int, nx::Int) where {S <: Real}
    flag = true
    γ = params.γ
    f̃k = out.f[k]
    out.errⱼ = estimate_excess(out.hj, k, f̃k, γ, nx)
    if out.errⱼ <= out.hj*params.tol
        out.predicted_hj = 0.9*out.hj*(0.5*out.hj/out.errⱼ)^(1/(k-1))
    else
        out.predicted_hj = out.hj*(out.hj*params.tol/out.errⱼ)^(1/(k-1))
        zjp1_temp = zjp1*(out.predicted_hj/out.hj)^k
        out.hj = out.predicted_hj
        zjp1 = zjp1_temp
        flag = false
    end
    flag
end

"""
$(TYPEDSIGNATURES)

Performs a single-step of the validated integrator. Input stepsize is out.step.
"""
function single_step!(out::StepResult{S}, params::StepParams, lf::LohnersFunctor,
                      stf!::TaylorFunctor!, Δ::CircularBuffer{Vector{S}},
                      A::CircularBuffer{QRDenseStorage}, P, rP) where {S <: Real}

    k = params.k

    # validate existence & uniqueness
    if ~out.jacobians_set
        jacobian_taylor_coeffs!(lf.jac_tf!, out.Xⱼ, P)
        extract_JxJp!(Jx, Jp, lf.jac_tf!.result, lf.jac_tf!.tjac, params.nx, np, k)
    end
    existence_uniqueness!(out, stf!, params.hmin, P)
    out.hj = out.unique_result.step
    if ~out.unique_result.confirmed
        out.status_flag = NUMERICAL_ERROR
        return nothing
    end

    # repeat until desired tolerance is otained or repetition limit is hit
    count = 1
    while (out.hj > params.hmin) && (count < params.repeat_limit)

        # perform corrector step
        out.status_flag = lf(out.hj, out.unique_result.X, out.Xⱼ, out.xⱼ, A, Δ, P, rP)

        # Perform Lepus error control scheme if step size not set
        if out.h > 0.0
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
    pushfirst!(Δ, get_Δ(lf))

    set_x!(out.xⱼ, lf)
    set_X!(out.Xⱼ, lf)
    nothing
end

function set_P!(d::DiscretizeRelax{X, IntervalArithmetic.Interval{Float64}}) where {X}
    @__dot__ d.P = Interval(d.pL, d.pU)
    @__dot__ d.rP = d.P - d.p
    nothing
end

function set_P!(d::DiscretizeRelax{X, MC{N,T}}) where {X, N, T<:RelaxTag}
    @__dot__ d.P = MC{N,NS}.(d.p, Interval(d.pL, d.pU), 1:np)
    @__dot__ d.rP = d.P - d.p
    nothing
end

function compute_X0!(d::DiscretizeRelax{X,T}) where {X, T <: Number}
    d.storage[:, 1] .= d.x0f(d.P)
    d.step_result.Xⱼ .= d.storage[:, 1]
    d.step_result.xⱼ .= mid.(d.step_result.Xⱼ)
    nothing
end

function set_Δ!(Δ::CircularBuffer{Vector{T}}, out::ElasticArray{T,2}) where T
    println("length(Δ): $(length(Δ))")
    println("out[:,1]: $(out[:,1])")
    Δ[1] = out[:,1]
    for i in 2:length(Δ)
        fill!(Δ[i], zero(T))
    end
    nothing
end

function DBB.relax!(d::DiscretizeRelax)

    set_P!(d)          # Functor set P and P - p values for calculations
    compute_X0!(d)     # Compute initial condition values

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
    println(" ")
    println("d.Δ: $(d.Δ)")
    set_Δ!(d.Δ, d.storage)
    println("d.Δ: $(d.Δ)")
    reinitialize!(d.A)

    # Begin integration loop
    hlast = 0.0
    d.step_result.hj = tmax - t
    d.step_count = 0
    while sign_tstep*t < sign_tstep*tmax

        # max step size is min of predicted, when next support point occurs,
        # or the last time step in the span
        d.step_result.hj = min(d.step_result.hj, next_support - t, tmax - t)

        # perform step size calculation and update bound information
        single_step!(d.step_result, d.step_params, d.method_f!, d.set_tf!, d.Δ, d.A, d.P, d.rP)

        # advance step counters
        t += d.step_result.hj
        d.step_count += 1

        # throw error if limit exceeded
        if d.step_count > d.step_limit
            d.error_code = LIMIT_EXCEEDED
            break
        elseif (d.step_result.status_flag !== RELAXATION_NOT_CALLED)
            d.error_code = d.step_result.status_flag
            break
        end

        # unpack storage
        if d.step_count > length(d.time)-1
            if d.nx == 1
                resize!(d.storage, 1, d.step_count*2)
            else
                resize!(d.storage, 2, d.step_count*2)
            end
            resize!(d.time, d.step_count*2)
        end
        d.storage[:, d.step_count+1] = d.step_result.Xⱼ
        d.time[d.step_count+1] = t
    end

    # cut out any unnecessary array elements
    if d.nx == 1
        resize!(d.storage, 1, d.step_count)
    else
        resize!(d.storage, 2, d.step_count)
    end
    resize!(d.time, d.step_count)
    if d.error_code === RELAXATION_NOT_CALLED
        d.error_code = COMPLETED
    end

    nothing
end
