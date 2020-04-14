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
    "Stores solution Y = (X,P) for each time"
    storage::ElasticArray{T,2}
    "Support index to storage dictory"
    support_dict::Dict{Int,Int}
    "Holds data for numeric error encountered in integration step"
    error_code::TerminationStatusCode
    "Storage for QR Factorizations"
    A::CircularBuffer{QRDenseStorage}
    "Storage for Δ"
    Δ::CircularBuffer{Vector{T}}
    "Relaxation Type"
    style::T

    # Main functions used in routines
    "Functor for evaluating Taylor coefficients over a set"
    set_tf!::TaylorFunctor!
    method_f!::LohnersFunctor

    step_result::StepResult{T}
    step_params::StepParams

end
function DiscretizeRelax(d::ODERelaxProb; repeat_limit = 50, step_limit = 1000,
                         tol = 1E-5, hmin = 1E-13, relax = false, k = 4,
                         method_steps = 2, γ = 1.0)
    nx = d.nx
    np = d.np
    p = d.p
    pL = d.pL
    pU = d.pU
    f = d.f

    tsupports = d.tsupports
    if ~isempty(tsupports)
        if (tsupports[1] == 0.0)
            support_dict = Dict{Int,Int}(d.support_dict, 1 => 1)
        end
    else
        support_dict = Dict{Int,Int}()
    end
    error_code = RELAXATION_NOT_CALLED

    T = relax ? MC{np,NS} : Interval{Float64}
    style = zero(T)
    storage = ElasticArray(zeros(T, nx+np, 1000))

    sizehint!(storage, nx+np, 10000)   # suggest 1000*(nx+np) steps likely
    A = qr_stack(nx, method_steps)
    Δ = CircularBuffer{Vector{T}}(method_steps)
    fill!(Δ, zeros(T, nx+np))

    set_tf! = TaylorFunctor!(f, nx, np, k, style, zero(Float64))
    method_f! = LohnersFunctor(f, nx, np, k, style, zero(Float64))

    step_result = StepResult(style, nx, np, k)
    step_params = StepParams(tol, hmin, nx, repeat_limit, γ, k)

    return DiscretizeRelax{typeof(d.x0),T}(d.x0, p, pL, pU, nx, np, d.tspan,
                                           d.tsupports, step_limit, 0, storage, support_dict,
                                           error_code, A, Δ, style, set_tf!,
                                           method_f!, step_result, step_params)
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

Performs a single-step of the validated integrator. Input stepsize is out.step.
"""
function single_step!(out::StepResult{S}, params::StepParams, lf::LohnersFunctor,
                      stf!::TaylorFunctor!, Δ::CircularBuffer{Vector{S}},
                      A::CircularBuffer{QRDenseStorage}) where {S <: Real}

    println(" ")
    println("single A: $(A)")
    println("single Δ: $(Δ)")

    k = params.k
    tol = params.tol
    γ = params.γ
    nx = params.nx
    hmin = params.hmin
    repeat_limit = params.repeat_limit

    # validate existence & uniqueness
    if ~out.jacobians_set
        jacobian_taylor_coeffs!(lf.jac_tf!, out.Yⱼ)
        extract_JxJp!(Jx, Jp, lf.jac_tf!.result, lf.jac_tf!.tjac, nx, np, k)
    end
    existence_uniqueness!(out, stf!, hmin)
    out.hj = out.unique_result.step
    if ~out.unique_result.confirmed
        out.status_flag = NUMERICAL_ERROR
        return nothing
    end

    # repeat until desired tolerance is otained or repetition limit is hit
    count = 1
    while (out.hj > hmin) && (count < repeat_limit)

        out.status_flag = lf(out.hj, out.unique_result.Y, out.Yⱼ, out.yⱼ, A, Δ)

        # Lepus error control scheme
        f̃k =
        out.errⱼ = estimate_excess(out.hj, k, f̃k, γ, nx)
        if out.errⱼ <= out.hj*params.tol
            out.predicted_hj = 0.9*out.hj*(0.5*out.hj/out.errⱼ)^(1/(k-1))
            break
        else
            out.predicted_hj = out.hj*(out.hj*tol/out.errⱼ)^(1/(k-1))
            zjp1_temp = zjp1*(out.predicted_hj/out.hj)^k
            out.hj = out.predicted_hj
            zjp1 = zjp1_temp
        end
        count += 1
    end

    # updates shifts Aj+1 -> Aj and so on
    pushfirst!(A, last(A))
    pushfirst!(Δ, get_Δ(lf))

    set_y!(out.yⱼ, lf)
    set_Y!(out.Yⱼ, lf)
    nothing
end

set_p!(d::DiscretizeRelax, p, pL, pU) = set_p!(d.method_f!, p, pL, pU)

function compute_y0!(out::ElasticArray{Interval{Float64},2}, x0, p, pL, pU, nx, np)
    P = Interval.(pL, pU)
    Xout = x0(P)
    out[1:nx, 1] .= Xout
    out[(nx+1):(nx+np), 1] .= P
    nothing
end

function compute_y0!(out::ElasticArray{MC{N,NS},2}, x0, p, pL, pU, nx, np) where N
    P = MC{N,NS}.(p, Interval.(pL, pU), 1:np)
    Xout = x0(P)
    copyto!(view(out, 1:nx, 1), 1, Xout, 1, nx)
    copyto!(view(out, (nx+1):(nx+np), 1), nx+1, P, 1, np)
    nothing
end

function set_Δ!(Δ::CircularBuffer{Vector{T}}, out::ElasticArray{T,2}) where T
    for i in 1:length(Δ)
        if i == 1
            Δ[1] .= out[:,1]
        else
            fill!(Δ[i], zero(T))
        end
    end
    nothing
end


function relax!(d::DiscretizeRelax)

    # Functor set P and P - p values for calculations
    set_p!(d.method_f!, d.p, d.pL, d.pU)

    # Compute initial condition values
    compute_y0!(d.storage, d.x0f, d.p, d.pL, d.pU, d.nx, d.np)
    d.step_result.Yⱼ .= d.storage[:, 1]
    d.step_result.yⱼ .= mid.(d.step_result.Yⱼ)

    # Get initial time and integration direction
    t = d.tspan[1]
    tmax = d.tspan[2]
    sign_tstep = copysign(1, tmax-t)

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
    println("d.A: $(d.A)")
    println("d.Δ: $(d.Δ)")
    set_Δ!(d.Δ, d.storage)
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
        single_step!(d.step_result, d.step_params, d.method_f!, d.set_tf!, d.Δ, d.A)

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
        if d.step_count > length(d.storage)
            resize!(d.storage, 2, d.step_count*2)
        else
            d.storage[:, step_count+1] .= d.step_result.Yⱼ
        end

    end

    if d.error_code === RELAXATION_NOT_CALLED
        d.error_code = COMPLETED
    end

    nothing
end
