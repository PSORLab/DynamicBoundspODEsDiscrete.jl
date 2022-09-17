
"""
$(TYPEDEF)
"""
abstract type AbstractIntervalCallback end

"""
$(TYPEDEF)

Functor object `d` that computes `h(xmid, P)` and `hj(X,P)` in-place 
using `X`, `P` information stored in the fields when `d()` is run. 
"""
mutable struct PICallback{FH,FJ} <:  AbstractIntervalCallback
    h!::FH
    hj!::FJ
    H::Vector{Interval{Float64}}
    J::Array{Interval{Float64},2}
    xmid::Vector{Float64}
    X::Vector{Interval{Float64}}
    P::Vector{Interval{Float64}}
    nx::Int
end
function PICallback(h!::FH, hj!::FJ, P::Vector{Interval{Float64}}, nx::Int) where {FH,FJ}
    H = zeros(Interval{Float64}, nx)
    J = zeros(Interval{Float64}, nx, nx)
    xmid = zeros(Float64, nx)
    X = zeros(Interval{Float64}, nx)
    return PICallback{FH,FJ}(h!, hj!, H, J, xmid, X, P, nx)
end

function (d::PICallback{FH,FJ})() where {FH,FJ}
    @unpack H, J, X, P, xmid, nx, h!, hj! = d
    fill!(H, zero(Interval{Float64}))
    fill!(J, zero(Interval{Float64}))
    for i in 1:nx
        xmid[i] = 0.5*(X[i].lo + X[i].hi)
    end
    h!(H, xmid, P)
    hj!(J, X, P)
    return
end

"""
$(TYPEDEF)
"""
function precondition!(d::DenseMidInv, H::Vector{Interval{Float64}}, J::Array{Interval{Float64},2})
    for i in eachindex(J)
        d.Y[i] = 0.5*(J[i].lo + J[i].hi)
    end
    F = lu!(d.Y)
    H .= F\H
    J .= F\J
    return
end

"""
$(TYPEDEF)
"""
abstract type AbstractContractor end

"""
$(TYPEDEF)
"""
@Base.kwdef struct NewtonInterval <: AbstractContractor
    N::Vector{Interval{Float64}}
    Ntemp::Vector{Interval{Float64}}
    X::Vector{Interval{Float64}}
    Xdiv::Vector{Interval{Float64}}
    inclusion::Vector{Bool}
    lower_inclusion::Vector{Bool}
    upper_inclusion::Vector{Bool}
    kmax::Int = 3 
    rtol::Float64 = 1E-6
    etol::Float64 = 1E-6
end
NewtonInterval(nx::Int) = NewtonInterval(N = zeros(Interval{Float64}, nx), 
                          Ntemp = zeros(Interval{Float64}, nx), 
                          X = zeros(Interval{Float64}, nx), 
                          Xdiv = zeros(Interval{Float64}, nx), 
                          inclusion = fill(false, (nx,)), 
                          lower_inclusion = fill(false, (nx,)),
                          upper_inclusion = fill(false, (nx,)))

function (d::NewtonInterval)(cb::PICallback{FH,FJ}) where {FH,FJ}
    @unpack X, Xdiv, N, Ntemp, inclusion, lower_inclusion, upper_inclusion, rtol = d
    @unpack H, J, nx = cb
    
    ext_division_flag = false
    exclusion_flag = false

    for i = 1:nx
        S1 = zero(Interval{Float64})
        S2 = zero(Interval{Float64})
        for j = 1:nx
            if j < i
                S1 += J[i,j]*(X[j] - 0.5*(X[j].lo + X[j].hi))
            elseif j > i
                S2 += J[i,j]*(X[j] - 0.5*(X[j].lo + X[j].hi))
            end
        end
        if J[i,i].lo*J[i,i].hi > 0.0
            N[i] = 0.5*(X[i].lo + X[i].hi) - (H[i] + S1 + S2)/J[i,i]
        else
            @. Ntemp = N
            eD, N[i], Ntemp[i] = extended_process(N[i], X[i], J[i,i], S1 + S2 + H[i], rtol)
            if isone(eD)
                ext_division_flag = true
                @. Xdiv = X
                Xdiv[i] = Ntemp[i] ∩ X[i]
                X[i] = N[i] ∩ X[i]
                return ext_division_flag, exclusion_flag
            end
        end
        if strict_x_in_y(N[i], X[i])
            inclusion[i] = true
            lower_inclusion[i] = true
            upper_inclusion[i] = true
        else
            inclusion[i] = false
            lower_inclusion[i] = N[i].lo > X[i].lo
            upper_inclusion[i] = N[i].hi < X[i].hi
        end
        if ~isdisjoint(N[i], X[i])
            X[i] = N[i] ∩ X[i]
        else
            return ext_division_flag, exclusion_flag
        end
    end
    return ext_division_flag, exclusion_flag
end

function parametric_interval_contractor(callback!::PICallback{FH,FJ}, precond!::P, contractor::S) where {FH, FJ, P, S <: AbstractContractor}

    exclusion_flag = false
    inclusion_flag = false
    ext_division_flag = false
    ext_division_num = 0
    fill!(contractor.inclusion, false)
    fill!(contractor.lower_inclusion, false)
    fill!(contractor.upper_inclusion, false)
    @. contractor.X = callback!.X

    for i = 1:contractor.kmax
        callback!()::Nothing
        precondition!(precond!, callback!.H, callback!.J)::Nothing
        exclusion_flag, ext_division_flag = contractor(callback!)::Tuple{Bool,Bool}
        (exclusion_flag || ext_division_flag) && break
        inclusion_flag = inclusion_test(inclusion_flag, contractor.inclusion, callback!.nx)
        @. callback!.X = contractor.X
        @. callback!.xmid = mid(callback!.X)
    end

    return exclusion_flag, inclusion_flag, ext_division_flag
end

"""
$(TYPEDEF)
"""
abstract type Wilhelm2019Type end
const W19T = Wilhelm2019Type

"""
$(TYPEDEF)

Use an implicit Euler style of relaxation.
"""
struct ImpEuler <: W19T end

"""
$(TYPEDEF)

Use an second-order Adam's Moulton method style of relaxation.
"""
struct AM2 <: W19T end

"""
$(TYPEDEF)

Use an second-order Backward Difference Formula method style of relaxation.
"""
struct BDF2 <: W19T end

state_contractor_integrator(m::ImpEuler) = ImplicitEuler(autodiff = false)
state_contractor_integrator(m::AM2) = Trapezoid(autodiff = false)
state_contractor_integrator(m::BDF2) = ABDF2(autodiff = false)


is_two_step(::ImpEuler) = false
is_two_step(::AM2) = false
is_two_step(::BDF2) = true

"""
$(TYPEDEF)

A callback function used for the Wilhelm2019 integrator.
"""
mutable struct CallbackH{V,F,T<:W19T} <: Function
    temp::Vector{V}
    xold1::Vector{V}
    xold2::Vector{V}
    h!::F
    t2::Float64
    t1::Float64
end
CallbackH{V,F,T}(nx::Int, h!::F) where {V,F,T<:W19T} = CallbackH{V,F,T}(zeros(V,nx), zeros(V,nx), zeros(V,nx), h!, 0.0, 0.0)
function (cb::CallbackH{V,F,ImpEuler})(out, x, p) where {V,F}
    @unpack h!, xold1, t1, t2 = cb
    h!(out, x, p, t2)
    @. out = out*(t2 - t1) - x + xold1
    nothing
end
function (cb::CallbackH{V,F,AM2})(out, x, p::Vector{V}) where {V,F}
    @unpack h!, temp, xold1, t1, t2 = cb
    h!(out, x, p, t2)
    h!(temp, xold1, p, t1)
    @. out = 0.5*(t2 - t1)*(out + temp) - x + xold1
    nothing
end
function (cb::CallbackH{V,F,BDF2})(out, x, p::Vector{V}) where {V,F}
    @unpack h!, xold1, xold2, t1, t2 = cb
    h!(out, x, p, t2)
    @. out = (2.0/3.0)*(t2 - t1)*out - x + (4.0/3.0)*xold1 - (1.0/3.0)*xold2
    nothing
end

"""
$(FUNCTIONNAME)

A callback function used for the Wilhelm2019 integrator.
"""
Base.@kwdef mutable struct CallbackHJ{F, T <: W19T} <: Function
    hj!::F
    tf::Float64 = 0.0
    delT::Float64 = 0.0
end
CallbackHJ{F,T}(hj!::F) where {F, T <: W19T} = CallbackHJ{F,T}(; hj! = hj!)

function (cb::CallbackHJ{F, ImpEuler})(out, x, p) where F
    @unpack hj!, tf, delT = cb
    hj!(out, x, p, tf)
    @. out *= delT
    for j in diagind(out)
        out[j] -= 1.0
    end
    nothing
end
function (cb::CallbackHJ{F, AM2})(out, x, p) where F
    @unpack hj!, tf, delT = cb 
    hj!(out, x, p, tf)
    @. out *= 0.5*delT
    for j in diagind(out)
        out[j] -= 1.0
    end
    nothing
end
function (cb::CallbackHJ{F, BDF2})(out, x, p) where F
    @unpack hj!, tf, delT = cb
    hj!(out, x, p, tf)
    @. out *= 2.0*delT/3.0
    for j in diagind(out)
        out[j] -= 1.0
    end
    nothing
end

"""
$(TYPEDEF)

An integrator that bounds the numerical solution of the pODEs system.

$(TYPEDFIELDS)
"""
mutable struct Wilhelm2019{T <: W19T, ICB1 <: PICallback, ICB2 <: PICallback, PRE, 
                           CTR <: AbstractContractor, IC <: Function, F, Z, J, PRB, N, C, AMAT} <: DBB.AbstractODERelaxIntegrator

    # problem specifications
    integrator_type::T
    time::Vector{Float64}
    steps::Int
    p::Vector{Float64}
    pL::Vector{Float64}
    pU::Vector{Float64}
    nx::Int
    np::Int
    xL::Array{Float64,2}
    xU::Array{Float64,2}

    # state of integrator flags
    integrator_states::IntegratorStates
    evaluate_interval::Bool
    evaluate_state::Bool
    differentiable_flag::Bool

    # storage used for parametric interval methods
    P::Vector{Interval{Float64}}
    X::Array{Interval{Float64},2}
    X0P::Vector{Interval{Float64}}
    pi_callback1::ICB1
    pi_callback2::ICB2
    pi_precond!::PRE
    pi_contractor::CTR
    inclusion_flag::Bool
    exclusion_flag::Bool
    extended_division_flag::Bool

    # callback functions used for MC methods
    ic::IC
    h1::CallbackH{Z,F,ImpEuler}
    h2::CallbackH{Z,F,T}
    hj1::CallbackHJ{J,ImpEuler}
    hj2::CallbackHJ{J,T}
    mccallback1::MCCallback
    mccallback2::MCCallback

    # storage used for MC methods
    IC_relax::Vector{Z}
    state_relax::Array{Z,2}
    var_relax::Vector{Z}
    param::Vector{Vector{Vector{Z}}}
    kmax::Int
    calculate_local_sensitivity::Bool

    # local evaluation information
    local_problem_storage
    prob
    constant_state_bounds::Union{Nothing,ConstantStateBounds}

    relax_t_dict_flt::Dict{Float64,Int}
    relax_t_dict_indx::Dict{Int,Int}
end

function Wilhelm2019(d::ODERelaxProb, t::T) where {T <: W19T}

    h! = d.f; hj! = d.Jx!
    time = d.support_set.s
    steps = length(time) - 1

    p = d.p; pL = d.pL; pU = d.pU
    nx = d.nx; np = length(p)
    xL = isempty(d.xL) ? zeros(nx,steps) : d.xL
    xU = isempty(d.xU) ? zeros(nx,steps) : d.xU

    P = Interval{Float64}.(pL, pU)
    X = zeros(Interval{Float64}, nx, steps)
    X0P = zeros(Interval{Float64}, nx)
    pi_precond! = DenseMidInv(zeros(Float64,nx,nx), zeros(Interval{Float64},1), nx, np)
    pi_contractor = NewtonInterval(nx)
    inclusion_flag = exclusion_flag = extended_division_flag = false
    Z = MC{np,NS}

    ic = d.x0
    h1 = CallbackH{Z,typeof(h!),ImpEuler}(nx, h!)
    h2 = CallbackH{Z,typeof(h!),T}(nx, h!)
    hj1 = CallbackHJ{typeof(hj!),ImpEuler}(hj!)
    hj2 = CallbackHJ{typeof(hj!),T}(hj!)

    h1intv! = CallbackH{Interval{Float64},typeof(h!),ImpEuler}(nx, h!)
    h2intv! = CallbackH{Interval{Float64},typeof(h!),T}(nx, h!)
    hj1intv! = CallbackHJ{typeof(hj!),ImpEuler}(hj!)
    hj2intv! = CallbackHJ{typeof(hj!),T}(hj!)

    pi_callback1 = PICallback(h1intv!, hj1intv!, P, nx)
    pi_callback2 = PICallback(h2intv!, hj2intv!, P, nx)
    mc_callback1 = MCCallback(h1, hj1, nx, np, McCormick.NewtonGS(), McCormick.DenseMidInv(zeros(nx,nx), zeros(Interval{Float64},1), nx, np))
    mc_callback2 = MCCallback(h2, hj2, nx, np, McCormick.NewtonGS(), McCormick.DenseMidInv(zeros(nx,nx), zeros(Interval{Float64},1), nx, np))

    # storage used for MC methods
    kmax = 1
    IC_relax = zeros(Z,nx)
    state_relax = zeros(Z, nx, steps)
    param = Vector{Vector{Z}}[[zeros(Z,nx) for j in 1:kmax] for i in 1:steps]
    var_relax = zeros(Z,np)

    calculate_local_sensitivity = true
    constant_state_bounds = d.constant_state_bounds

    local_integrator() = state_contractor_integrator(t)
    local_problem_storage = ODELocalIntegrator(d, local_integrator)

    support_set = DBB.get(d, DBB.SupportSet())
    relax_t_dict_flt = Dict{Float64,Int}()
    relax_t_dict_indx = Dict{Int,Int}()
    for (i,s) in enumerate(support_set.s)
        relax_t_dict_flt[s] = i
        relax_t_dict_indx[i] = i
    end

    return Wilhelm2019{T, typeof(pi_callback1), typeof(pi_callback2),
                       typeof(pi_precond!), typeof(pi_contractor),
                       typeof(ic), typeof(h!), Z, typeof(hj!), nothing,
                       np, NewtonGS, Array{Float64,2}}(t, time, steps, p, pL, pU, nx, np, xL, xU,
                       IntegratorStates(), false, false, false,
                       P, X, X0P, pi_callback1, pi_callback2, pi_precond!,
                       pi_contractor, inclusion_flag, exclusion_flag,
                       extended_division_flag, ic, h1, h2, hj1, hj2,
                       mc_callback1, mc_callback2, IC_relax, state_relax,
                       var_relax, param, kmax, calculate_local_sensitivity, 
                       local_problem_storage, d, constant_state_bounds,
                       relax_t_dict_flt, relax_t_dict_indx)
end

function get_val_loc(d::Wilhelm2019, i::Int, t::Float64)
    (i <= 0 && t == -Inf) && error("Must set either index or time.")
    (i > 0) ? d.relax_t_dict_indx[i] : d.relax_t_dict_flt[t]
end

is_new_box(d::Wilhelm2019) = d.integrator_states.new_decision_box
use_relax(d::Wilhelm2019) = !d.evaluate_interval
use_relax_new_pnt(d::Wilhelm2019) = d.integrator_states.new_decision_pnt && use_relax(d)

function relax!(d::Wilhelm2019{T, ICB1, ICB2, PRE, CTR, IC, F, Z, J, PRB, N, C, AMAT}) where {T, ICB1, ICB2, PRE, CTR, IC, F, Z, J, PRB, N, C, AMAT}
    pi_cb1 = d.pi_callback1; pi_cb2 = d.pi_callback2
    mc_cb1 = d.mccallback1;  mc_cb2 = d.mccallback2

    # load state relaxation bounds at support values
    if !isnothing(d.constant_state_bounds)
        for i = 1:d.steps
            d.X[:,i] .= Interval{Float64}.(d.constant_state_bounds.xL, d.constant_state_bounds.xU)
        end
    end

    if is_two_step(d.integrator_type)
        if is_new_box(d)
            d.X0P .= d.ic(d.P)                   # evaluate initial condition

            # loads CallbackH and CallbackHJ function with correct time and prior x values
            @. pi_cb1.X = d.X[:,1]
            pi_cb1.xmid .= mid.(pi_cb1.X)
            pi_cb1.h!.xold1 .= d.X0P
            pi_cb1.h!.t1 = 0.0
            pi_cb1.h!.t2 = pi_cb1.hj!.tf = pi_cb1.hj!.delT = d.time[2]
            @. pi_cb1.P = pi_cb2.P = d.P
            
            # run interval contractor on first step & break if solution is proven not to exist
            excl, incl, extd = parametric_interval_contractor(pi_cb1, d.pi_precond!, d.pi_contractor)
            excl && (d.integrator_states.termination_status = EMPTY; return)
            d.exclusion_flag = excl
            d.inclusion_flag = incl
            d.extended_division_flag = extd
            @. d.X[:,1] = d.pi_contractor.X    # store interval values to storage array in d

            # generate reference point relaxations
            if use_relax(d)
                for j=1:d.np
                    p_mid = 0.5*(lo(d.P[j]) + hi(d.P[j]))
                    mc_cb1.pref_mc[j] = MC{d.np,NS}(p_mid, d.P[j], j)
                end
                @. mc_cb1.P = mc_cb2.P = d.P
                @. mc_cb1.p_mc = mc_cb1.pref_mc
                @. mc_cb2.p_mc = mc_cb1.pref_mc
                @. mc_cb2.pref_mc = mc_cb1.pref_mc
                @. mc_cb1.X = d.X[:,1]

                # evaluate initial condition
                d.IC_relax .= d.ic(mc_cb1.pref_mc)

                # loads CallbackH and CallbackHJ function with correct time and prior x values
                mc_cb1.h!.t1 = 0.0
                mc_cb1.h!.t2 = mc_cb1.hj!.tf = mc_cb1.hj!.delT = d.time[2]
                @. mc_cb1.h!.xold1 = d.IC_relax

                # generate and save reference point relaxations
                gen_expansion_params!(mc_cb1)
                for q = 1:d.kmax
                    @. d.param[1][q] = mc_cb1.param[q]
                end

                # generate and save relaxation at reference point
                implicit_relax_h!(mc_cb1)
                @. d.state_relax[:,1] = mc_cb1.x_mc
            end
        end

        if use_relax_new_pnt(d)
            for j = 1:d.np
                d.var_relax[j] = MC{d.np,NS}(d.p[j], d.P[j], j)
            end
            d.IC_relax .= d.ic(d.var_relax)

            # loads MC callback, CallbackH and CallbackHJ function with correct time and prior x values
            @. mc_cb1.p_mc = mc_cb2.p_mc = d.var_relax
            @. mc_cb1.X = d.X[:,1]
            mc_cb1.h!.t1 = 0.0
            mc_cb1.h!.t2 = mc_cb1.hj!.tf = mc_cb1.hj!.delT = d.time[2]
            mc_cb1.h!.xold1 .= d.IC_relax
            for q = 1:d.kmax
                @. mc_cb1.param[q] = d.param[1][q]
            end
            implicit_relax_h!(mc_cb1) # computes relaxation

            @. d.state_relax[:,1] = mc_cb1.x_mc
        end
    else
        if is_new_box(d)
            d.X0P .= d.ic(d.P)                   # evaluate initial condition
            # loads CallbackH and CallbackHJ function with correct time and prior x values
            @. pi_cb2.X = d.X[:,1]
            pi_cb2.xmid .= mid.(pi_cb1.X)
            pi_cb2.h!.xold1 .= d.X0P
            pi_cb2.h!.t1 = 0.0
            pi_cb2.h!.t2 = pi_cb2.hj!.tf = pi_cb2.hj!.delT = d.time[2]
            @. pi_cb2.P = pi_cb2.P = d.P
            
            # run interval contractor on first step & break if solution is proven not to exist
            excl, incl, extd = parametric_interval_contractor(pi_cb2, d.pi_precond!, d.pi_contractor)
            excl && (d.integrator_states.termination_status = EMPTY; return)
            d.exclusion_flag = excl
            d.inclusion_flag = incl
            d.extended_division_flag = extd
            @. d.X[:,1] = d.pi_contractor.X    # store interval values to storage array in d

            # generate reference point relaxations
            if use_relax(d)
                for j=1:d.np
                    p_mid = 0.5*(lo(d.P[j]) + hi(d.P[j]))
                    mc_cb2.pref_mc[j] = MC{d.np,NS}(p_mid, d.P[j], j)
                end
                @. mc_cb2.P = mc_cb2.P = d.P
                @. mc_cb2.p_mc = mc_cb2.pref_mc
                @. mc_cb2.p_mc = mc_cb2.pref_mc
                @. mc_cb2.pref_mc = mc_cb2.pref_mc
                @. mc_cb2.X = d.X[:,1]

                # evaluate initial condition
                d.IC_relax .= d.ic(mc_cb2.pref_mc)

                # loads CallbackH and CallbackHJ function with correct time and prior x values
                mc_cb2.h!.t1 = 0.0
                mc_cb2.h!.t2 = mc_cb2.hj!.tf = mc_cb2.hj!.delT = d.time[2]
                @. mc_cb2.h!.xold1 = d.IC_relax

                # generate and save reference point relaxations
                gen_expansion_params!(mc_cb2)
                for q = 1:d.kmax
                    @. d.param[1][q] = mc_cb2.param[q]
                end

                # generate and save relaxation at reference point
                implicit_relax_h!(mc_cb2)
                @. d.state_relax[:,1] = mc_cb2.x_mc
                subgradient_expansion_interval_contract!(d.state_relax[:,1], d.p, d.pL, d.pU)
                @. d.X[:,1] = Intv(d.state_relax[:,1])
            end
        end

        if use_relax_new_pnt(d)
            for j = 1:d.np
                d.var_relax[j] = MC{d.np,NS}(d.p[j], d.P[j], j)
            end
            d.IC_relax .= d.ic(d.var_relax)

            # loads MC callback, CallbackH and CallbackHJ function with correct time and prior x values
            @. mc_cb2.p_mc = mc_cb2.p_mc = d.var_relax
            @. mc_cb2.X = d.X[:,1]
            mc_cb2.h!.t1 = 0.0
            mc_cb2.h!.t2 = mc_cb2.hj!.tf = mc_cb2.hj!.delT = d.time[2]
            mc_cb2.h!.xold1 .= d.IC_relax
            for q = 1:d.kmax
                @. mc_cb2.param[q] = d.param[1][q]
            end
            implicit_relax_h!(mc_cb2) # computes relaxation

            @. d.state_relax[:,1] = mc_cb2.x_mc
        end
    end

    for i in 2:d.steps
        if is_new_box(d)
            @. pi_cb2.X = d.X[:, i]          # load CallbackH and CallbackHJ with time and prior x
            @. pi_cb2.h!.xold1 = d.X[:, i-1]
            if i == 2
                @. pi_cb2.h!.xold2 = d.X0P
            else
                @. pi_cb2.h!.xold2 = d.X[:,i-2]
            end
            pi_cb2.h!.t1 = d.time[i]
            pi_cb2.h!.t2 = pi_cb2.hj!.tf = d.time[i + 1]
            pi_cb2.hj!.delT = d.time[i + 1] - d.time[i]

            # run interval contractor on ith step
            excl, incl, extd = parametric_interval_contractor(pi_cb2, d.pi_precond!, d.pi_contractor)
            excl && (d.integrator_states.termination_status = EMPTY; return)
            d.exclusion_flag = excl
            d.inclusion_flag = incl
            d.extended_division_flag = extd
            @. d.X[:,i] = d.pi_contractor.X

            if use_relax(d)
                # loads CallbackH and CallbackHJ function with correct time and prior x values
                mc_cb1.h!.t1 = 0.0
                mc_cb1.h!.t2 = mc_cb1.hj!.tf = mc_cb1.hj!.delT = d.time[2]
                @. mc_cb1.h!.xold1 = d.IC_relax
    
                # generate and save reference point relaxations
                gen_expansion_params!(mc_cb1)
                for q = 1:d.kmax
                    @. d.param[i][q] = mc_cb1.param[q]
                end
    
                # generate and save relaxation at reference point
                implicit_relax_h!(mc_cb1)
                @. d.state_relax[:,i] = mc_cb1.x_mc
                subgradient_expansion_interval_contract!(d.state_relax[:,i], d.p, d.pL, d.pU)
                @. d.X[:,i] = Intv(d.state_relax[:,i])
                # update interval bounds for state relaxation...
            end
        end
        if use_relax_new_pnt(d)
            # loads MC callback, CallbackH and CallbackHJ with correct time and prior x values
            @. mc_cb2.X = d.X[:,i]
            @. mc_cb2.h!.xold1 = d.state_relax[:, i-1]
            if i == 2
                @. mc_cb2.h!.xold2 = d.IC_relax
            else
                @. mc_cb2.h!.xold2 = d.state_relax[:,i-2]
            end
            mc_cb2.h!.t1 = d.time[i]
            mc_cb2.h!.t2 = mc_cb2.hj!.tf = d.time[i+1]
            mc_cb2.hj!.delT = d.time[i+1] - d.time[i]
           
            for q = 1:d.kmax
                @. mc_cb2.param[q] = d.param[i][q]
            end

            # computes relaxation
            implicit_relax_h!(mc_cb2)
            @. d.state_relax[:, i] = mc_cb2.x_mc
        end 
    end

    # unpack interval bounds to integrator bounds & set evaluation flags
    if !use_relax(d)
        map!(lo, d.xL, d.X)
        map!(hi, d.xU, d.X)
    else
        map!(lo, d.xL, d.state_relax)
        map!(hi, d.xU, d.state_relax)
    end
    d.integrator_states.new_decision_box = false
    d.integrator_states.new_decision_pnt = false
    return
end

function DBB.integrate!(d::Wilhelm2019, p::ODERelaxProb)

    local_integrator() = state_contractor_integrator(d.integrator_type)
    local_prob_storage = DBB.get(d, DBB.LocalIntegrator())::ODELocalIntegrator
    local_prob_storage.integrator = local_integrator
    local_prob_storage.adaptive_solver = false
    local_prob_storage.user_t = d.time

    DBB.getall!(local_prob_storage.p, d, ParameterValue())
    local_prob_storage.pduals .= DBB.seed_duals(Val(length(local_prob_storage.p)), local_prob_storage.p)
    local_prob_storage.x0duals = p.x0(d.local_problem_storage.pduals)
    solution_t = DBB.integrate!(Val(DBB.get(d, DBB.LocalSensitivityOn())), d, p)

    empty!(local_prob_storage.local_t_dict_flt)
    empty!(local_prob_storage.local_t_dict_indx)

    for (tindx, t) in enumerate(solution_t)
        local_prob_storage.local_t_dict_flt[t] = tindx
    end

    if !isempty(local_prob_storage.user_t)
        next_support_time = local_prob_storage.user_t[1]
        supports_left = length(local_prob_storage.user_t)
        loc_count = 1
        for (tindx, t) in enumerate(solution_t)
            if t == next_support_time
                local_prob_storage.local_t_dict_indx[loc_count] = tindx
                loc_count += 1
                supports_left -= 1
                if supports_left > 0
                    next_support_time = local_prob_storage.user_t[loc_count]
                end
            end
        end
    end
    return
end

DBB.supports(::Wilhelm2019, ::DBB.IntegratorName) = true
DBB.supports(::Wilhelm2019, ::DBB.Gradient) = true
DBB.supports(::Wilhelm2019, ::DBB.Subgradient) = true
DBB.supports(::Wilhelm2019, ::DBB.Bound) = true
DBB.supports(::Wilhelm2019, ::DBB.Relaxation) = true
DBB.supports(::Wilhelm2019, ::DBB.IsNumeric) = true
DBB.supports(::Wilhelm2019, ::DBB.IsSolutionSet) = true
DBB.supports(::Wilhelm2019, ::DBB.TerminationStatus) = true
DBB.supports(::Wilhelm2019, ::DBB.Value) = true
DBB.supports(::Wilhelm2019, ::DBB.ParameterValue) = true
DBB.supports(::Wilhelm2019, ::DBB.ConstantStateBounds) = true

DBB.get(t::Wilhelm2019, v::DBB.IntegratorName) = "Wilhelm 2019 Integrator"
DBB.get(t::Wilhelm2019, v::DBB.IsNumeric) = true
DBB.get(t::Wilhelm2019, v::DBB.IsSolutionSet) = false
DBB.get(t::Wilhelm2019, s::DBB.TerminationStatus) = t.integrator_states.termination_status

DBB.get(t::Wilhelm2019, s::DBB.ParameterNumber) = t.np
DBB.get(t::Wilhelm2019, s::DBB.StateNumber) = t.nx
DBB.get(t::Wilhelm2019, s::DBB.SupportNumber) = length(t.time)
DBB.get(t::Wilhelm2019, s::DBB.AttachedProblem) = t.prob


function DBB.set!(t::Wilhelm2019, v::ConstantStateBounds)
    t.constant_state_bounds = v
    return
end

function DBB.get(t::Wilhelm2019, v::DBB.LocalIntegrator)
    return t.local_problem_storage
end

function DBB.getall!(out::Array{Float64,2}, t::Wilhelm2019, v::DBB.Value)
    out .= t.local_problem_storage.pode_x
    return
end

function DBB.getall!(out::Vector{Array{Float64,2}}, t::Wilhelm2019, g::DBB.Gradient{Lower})
    if ~t.differentiable_flag
        error("Integrator does not generate differential relaxations. Set the
               differentiable_flag field to true and reintegrate.")
    end
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            for j in eachindex(out[i])
                out[i][j] = t.state_relax[j].cv_grad[i]
            end
        end
    end
    return
end
function DBB.getall!(out::Vector{Array{Float64,2}}, t::Wilhelm2019, g::DBB.Gradient{Upper})
    if ~t.differentiable_flag
        error("Integrator does not generate differential relaxations. Set the
               differentiable_flag field to true and reintegrate.")
    end
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.state_relax[j].cc_grad[i]
            end
        end
    end
    return
end

function DBB.getall!(out::Vector{Array{Float64,2}}, t::Wilhelm2019, g::DBB.Subgradient{Lower})
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.state_relax[j].cv_grad[i]
            end
        end
    end
    return
end
function DBB.getall!(out::Vector{Array{Float64,2}}, t::Wilhelm2019, g::DBB.Subgradient{Upper})
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.state_relax[j].cc_grad[i]
            end
        end
    end
    return
end

function DBB.getall!(out::Array{Float64,2}, t::Wilhelm2019, v::DBB.Bound{Lower})
    for i in 1:t.nx
        out[i,1] = t.X0P[i].lo
    end
    out[:,2:end] .= t.xL
    return
end

function DBB.getall!(out::Vector{Float64}, t::Wilhelm2019, v::DBB.Bound{Lower})
    out[:] = t.xL[1,:]
    return
end

function DBB.getall!(out::Array{Float64,2}, t::Wilhelm2019, v::DBB.Bound{Upper})
    for i in 1:t.nx
        out[i,1] = t.X0P[i].hi
    end
    out[:,2:end] .= t.xU
    return
end

function DBB.getall!(out::Vector{Float64}, t::Wilhelm2019, v::DBB.Bound{Upper})
    out[:] = t.xU[1,:]
    return
end

function DBB.getall!(out::Array{Float64,2}, t::Wilhelm2019, v::DBB.Relaxation{Lower})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].lo
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cv
        end
    end
    return
end
function DBB.getall!(out::Vector{Float64}, t::Wilhelm2019, v::DBB.Relaxation{Lower})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].lo
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cv
        end
    end
    return
end

function DBB.getall!(out::Array{Float64,2}, t::Wilhelm2019, v::DBB.Relaxation{Upper})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].hi
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cc
        end
    end
    return
end
function DBB.getall!(out::Vector{Float64}, t::Wilhelm2019, v::DBB.Relaxation{Upper})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].hi
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cc
        end
    end
    return
end

function DBB.getall!(t::Wilhelm2019, v::DBB.ParameterBound{Lower})
    @inbounds for i in 1:t.np
        out[i] = t.pL[i]
    end
    return
end
DBB.getall(t::Wilhelm2019, v::DBB.ParameterBound{Lower}) = t.pL

function DBB.getall!(out, t::Wilhelm2019, v::DBB.ParameterBound{Upper})
    @inbounds for i in 1:t.np
        out[i] = t.pU[i]
    end
    return
end
DBB.getall(t::Wilhelm2019, v::DBB.ParameterBound{Upper}) = t.pU

function DBB.setall!(t::Wilhelm2019, v::DBB.ParameterBound{Lower}, value::Vector{Float64})
    t.integrator_states.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pL[i] = value[i]
    end
    return
end

function DBB.setall!(t::Wilhelm2019, v::DBB.ParameterBound{Upper}, value::Vector{Float64})
    t.integrator_states.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pU[i] = value[i]
    end
    return
end

function DBB.setall!(t::Wilhelm2019, v::DBB.ParameterValue, value::Vector{Float64})
    t.integrator_states.new_decision_pnt = true
    @inbounds for i in 1:t.np
        t.p[i] = value[i]
    end
    return
end

function DBB.getall!(out, t::Wilhelm2019, v::DBB.ParameterValue)
    @inbounds for i in 1:t.np
        out[i] = t.p[i]
    end
    return
end

function DBB.setall!(t::Wilhelm2019, v::DBB.Bound{Lower}, values::Array{Float64,2})
    if t.integrator_states.new_decision_box
        t.integrator_states.set_lower_state = true
    end
    for i in 1:t.nx
        @inbounds for j in 1:t.steps
            t.xL[i,j] = values[i,j]
        end
    end
    return
end

function DBB.setall!(t::Wilhelm2019, v::DBB.Bound{Lower}, values::Vector{Float64})
    if t.integrator_states.new_decision_box
        t.integrator_states.set_lower_state = true
    end
    @inbounds for i in 1:t.steps
        t.xL[1,i] = values[i]
    end
    return
end

function DBB.setall!(t::Wilhelm2019, v::DBB.Bound{Upper}, values::Array{Float64,2})
    if t.integrator_states.new_decision_box
        t.integrator_states.set_upper_state = true
    end
    for i in 1:t.nx
        @inbounds for j in 1:t.steps
            t.xU[i,j] = values[i,j]
        end
    end
    return
end

function DBB.setall!(t::Wilhelm2019, v::DBB.Bound{Upper}, values::Vector{Float64})
    if t.integrator_states.new_decision_box
        t.integrator_states.set_upper_state = true
    end
    @inbounds for i in 1:t.steps
        t.xU[1,i] = values[i]
    end
    return
end

DBB.get(t::Wilhelm2019, v::DBB.SupportSet{T}) where T = DBB.get(t.prob, v)
DBB.get(t::Wilhelm2019, v::DBB.LocalSensitivityOn) = t.calculate_local_sensitivity
function DBB.set!(t::Wilhelm2019, v::DBB.LocalSensitivityOn, b::Bool) 
    t.calculate_local_sensitivity = b
end

function DBB.get!(out, t::Wilhelm2019, v::DBB.Bound{Lower})
    vi = get_val_loc(t, v.index, v.time)
    if vi <= 1
        return t.evaluate_interval ? map!(lo, out, t.X0P) : map!(lo, out, t.IC_relax)
    end
    if t.evaluate_interval
        out .= view(t.xL, :, vi - 1)
    else
        map!(lo, out, view(t.state_relax, :, vi - 1))
    end
end
function DBB.get!(out, t::Wilhelm2019, v::DBB.Bound{Upper})
    vi = get_val_loc(t, v.index, v.time)
    if vi <= 1
        return t.evaluate_interval ? map!(hi, out, t.X0P) : map!(hi, out, t.IC_relax)
    end
    if t.evaluate_interval
        out .= view(t.xU, :, vi - 1)
    else
        map!(hi, out, view(t.state_relax, :, vi - 1))
    end
end
function DBB.get!(out, t::Wilhelm2019, v::DBB.Relaxation{Lower})
    vi = get_val_loc(t, v.index, v.time)
    if vi <= 1
        return t.evaluate_interval ? map!(lo, out, t.X0P) : map!(cv, out, t.IC_relax)
    end
    if t.evaluate_interval
        out .= view(t.xL, :, vi - 1)
    else
        map!(cv, out, view(t.state_relax, :, vi - 1))
    end
end
function DBB.get!(out, t::Wilhelm2019, v::DBB.Relaxation{Upper})
    vi = get_val_loc(t, v.index, v.time)
    if vi <= 1
        return t.evaluate_interval ? map!(hi, out, t.X0P) : map!(cc, out, t.IC_relax)
    end
    if t.evaluate_interval
        out .= view(t.xU, :, vi - 1)
    else
        map!(cc, out, view(t.state_relax, :, vi - 1))
    end
end
function DBB.get!(out, t::Wilhelm2019, v::DBB.Subgradient{Lower})
    vi = get_val_loc(t, v.index, v.time)
    t.evaluate_interval && (return fill!(out, 0.0);)
    ni, nj = size(out)
    if vi <= 1
        for i = 1:ni, j = 1:nj
            out[i, j] = t.IC_relax[i].cv_grad[j]
        end
    else
        for i = 1:ni, j = 1:nj
            out[i, j] = t.state_relax[i,vi-1].cv_grad[j]
        end
    end
end
function DBB.get!(out, t::Wilhelm2019, v::DBB.Subgradient{Upper})
    vi = get_val_loc(t, v.index, v.time)
    t.evaluate_interval && (return fill!(out, 0.0);)
    ni, nj = size(out)
    if vi <= 1
        for i = 1:ni, j = 1:nj
            out[i, j] = t.IC_relax[i].cc_grad[j]
        end
    else
        for i = 1:ni, j = 1:nj
            out[i, j] = t.state_relax[i,vi-1].cc_grad[j]
        end
    end
end

"""
$(FUNCTIONNAME)

Returns true if X1 and X2 are equal to within tolerance atol in all dimensions.
"""
function is_equal(X1::S, X2::Vector{Interval{Float64}}, atol::Float64, nx::Int) where S
    out::Bool = true
    @inbounds for i=1:nx
        if (abs(X1[i].lo - X2[i].lo) >= atol ||
            abs(X1[i].hi - X2[i].hi) >= atol )
            out = false
            break
        end
    end
    return out
end

"""
$(FUNCTIONNAME)

Returns true if X is strictly in Y (X.lo>Y.lo && X.hi<Y.hi).
"""
function strict_x_in_y(X::Interval{Float64}, Y::Interval{Float64})
  (X.lo <= Y.lo) && return false
  (X.hi >= Y.hi) && return false
  return true
end

"""
$(FUNCTIONNAME)
"""
function inclusion_test(inclusion_flag::Bool, inclusion_vector::Vector{Bool}, nx::Int)
    if !inclusion_flag
        for i=1:nx
            if @inbounds inclusion_vector[i]
                inclusion_flag = true
            else
                inclusion_flag = false; break
            end
        end
    end
    return inclusion_flag
end

"""
$(FUNCTIONNAME)

Subfunction to generate output for extended division.
"""
function extended_divide(A::Interval{Float64})
    if (A.lo == -0.0) && (A.hi == 0.0)
        B::Interval{Float64} = Interval{Float64}(-Inf,Inf)
        C::Interval{Float64} = B
        return 0,B,C
    elseif (A.lo == 0.0)
        B = Interval{Float64}(1.0/A.hi,Inf)
        C = Interval{Float64}(Inf,Inf)
        return 1,B,C
    elseif (A.hi == 0.0)
        B = Interval{Float64}(-Inf,1.0/A.lo)
        C = Interval{Float64}(-Inf,-Inf)
        return 2,B,C
    else
        B = Interval{Float64}(-Inf,1.0/A.lo)
        C = Interval{Float64}(1.0/A.hi,Inf)
        return 3,B,C
    end
end

"""
$(FUNCTIONNAME)

Generates output boxes for extended division and flag.
"""
function extended_process(N::Interval{Float64}, X::Interval{Float64},
                          Mii::Interval{Float64}, SB::Interval{Float64},
                          rtol::Float64)

    Ntemp = Interval{Float64}(N.lo, N.hi)
    M = SB + Interval{Float64}(-rtol, rtol)
    if (M.lo <= 0) && (M.hi >= 0)
        return 0, Interval{Float64}(-Inf,Inf), Ntemp
    end

    k, IML, IMR = extended_divide(Mii)
    if (k === 1)
        NL = 0.5*(X.lo+X.hi) - M*IML
        return 0, NL, Ntemp
    elseif (k === 2)
        NR = 0.5*(X.lo+X.hi) - M*IMR
        return 0, NR, Ntemp
    elseif (k === 3)
        NR = 0.5*(X.lo+X.hi) - M*IMR
        NL = 0.5*(X.lo+X.hi) - M*IML
        if ~isdisjoint(NL,X) && isdisjoint(NR,X)
            return 0, NL, Ntemp
        elseif ~isdisjoint(NR,X) && isdisjoint(NL,X)
            return 0, NR, Ntemp
        elseif ~isdisjoint(NL,X) && ~isdisjoint(NR,X)
            N = NL
            Ntemp = NR
            return 1, NL, NR
        else
            return -1, N, Ntemp
        end
    end
    return 0, N, Ntemp
end
