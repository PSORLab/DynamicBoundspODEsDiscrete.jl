# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# Dynamic Bounds - pODEs Discrete
# A package for discretize and relax methods for bounding pODEs.
# See https://github.com/PSORLab/DynamicBoundspODEsDiscrete.jl
#############################################################################
# src/DiscretizeRelax/utilities/relax.jl
# Defines the relax! routine for the DiscretizeRelax integrator.
#############################################################################

fill_zero!(x::Vector{T}) where T = fill!(x, zero(T))
function fill_identity!(x::Vector{T}) where T
    x[1] = one(T)
end
function fill_identity!(x::Matrix{T}) where T
    fill!(x, zero(T))
    nx = size(x, 1)
    for i = 1:nx
        x[i,i] = one(T)
    end
end

function populate_storage_buffer!(z::Vector{T}, nt) where T
    foreach(i -> push!(z, zero(T)), length(z):nt)
    fill!(z, zero(T))
end
function populate_storage_buffer!(z::Vector{Vector{T}}, nt, nx) where T
    foreach(i -> push!(z, zeros(T, nx)), length(z):nt)
    foreach(fill_zero!, z)
end

# GOOD
"""
    reset_relax!

AAA
"""
function initialize_relax!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY}

    @unpack time, storage, storage_apriori, storage_buffer_size, tsupports, nx, tspan = d
    @unpack A_Q, A_inv, Δ = d.contractor_result

    # reset apriori and contracted set storage along with time storage
    populate_storage_buffer!(storage_apriori, storage_buffer_size, nx)
    populate_storage_buffer!(storage, storage_buffer_size, nx)
    populate_storage_buffer!(time, storage_buffer_size)

    # reset dictionarys used to map storage to supported points
    empty!(d.relax_t_dict_indx)
    empty!(d.relax_t_dict_flt)

    # reset buffers used to hold parallelpepid representation state bound/relaxation
    foreach(fill_identity!, A_Q)
    foreach(fill_identity!, A_inv)
    set_Δ!(Δ, storage)

    d.step_count = 1
    if length(tsupports) > 0
        d.next_support_i = 1
        if tsupports[1] > 0.0 
            d.next_support = tsupports[1]
        elseif length(tsupports) > 1
            d.next_support = tsupports[2]
        end
    else
        d.next_support_i = 1E10 
        d.next_support = Inf
    end

    ex = d.exist_result
    ex.hj = !(ex.hj <= 0.0) ? ex.hj : 0.05*(tspan[2] - d.contractor_result.times[1])
    ex.predicted_hj = ex.hj
    d.contractor_result.hj = ex.hj

    d.step_result.time = 0.0
    d.time[1] = tspan[1]
    d.contractor_result.times[1] = tspan[1]

    set_P!(d)

    # reset result flag
    d.exist_result.status_flag = RELAXATION_NOT_CALLED
    return
end

function set_starting_existence_bounds!(ex::ExistStorage{F,K,S,T}, c::ContractorStorage{T}, r::StepResult{T}) where {F,K,S,T}
    ex.Xj_0 .= r.Xⱼ
    ex.predicted_hj = c.hj
    nothing
end

function set_result_info!(r, hj, predicted_hj)
    r.time = round(r.time + hj, digits=13)
    r.predicted_hj = predicted_hj
    nothing
end

"""
    store_step_result!

Store result from step contractor to storage for state relaxation, apriori, times. Sets the dicts and updates the step count.    
"""
function store_step_result!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY}
    @unpack time, storage, storage_apriori, storage_buffer_size, tsupports, step_count, nx, step_result = d
    @unpack X_computed, hj = d.contractor_result
    @unpack Xj_apriori = d.exist_result

    set_result_info!(step_result, hj, hj)  # add time and predicted step size to results
    set_starting_existence_bounds!(d.exist_result, d.contractor_result, step_result)

    if step_count + 1 > length(time)
        push!(storage, copy(X_computed))
        push!(storage_apriori, copy(Xj_apriori))
        push!(time, d.step_result.time)
    else
        storage_position = step_count + 1
        copyto!(storage[storage_position], X_computed)
        copyto!(storage_apriori[storage_position], Xj_apriori)
        time[storage_position] = d.step_result.time
    end

    if d.step_result.time == d.next_support
        d.relax_t_dict_indx[d.next_support_i] = d.step_count
        d.relax_t_dict_flt[d.next_support] = d.step_count

        if d.next_support_i <= length(tsupports)
            d.next_support_i += 1 
            if (0.0 in tsupports)
                if d.next_support_i < length(tsupports)
                    d.next_support = tsupports[d.next_support_i + 1]
                else
                    d.next_support_i = typemax(Int)
                    d.next_support = Inf
                end
            else
                d.next_support = tsupports[d.next_support_i]
            end
        else
            d.next_support_i = typemax(Int)
            d.next_support = Inf
        end
    end

    d.step_count += 1

    return
end

"""
    clean_results!

Resize storage at the end to eliminate any unused values. If no error is set, record the error as COMPLETED.
"""
function clean_results!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY}
    @unpack step_count, storage, storage_apriori, time = d

    resize!(storage, step_count)
    resize!(storage_apriori, step_count)
    resize!(time, step_count)
    if d.error_code == RELAXATION_NOT_CALLED
        d.error_code = COMPLETED
    end

    nothing
end

"""
    relax_loop_terminated!

Checks for termination at the start of each step. An error code is stored the limit is exceeded.
"""
function continue_relax_loop!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M,T,S,F,K,X,NY}
    @unpack step_count, step_limit, time, tspan = d
    @unpack predicted_hj, status_flag = d.exist_result

    should_continue = true
    sign_tstep = copysign(1, tspan[2] - d.step_result.time)
    if step_count >= step_limit
        if sign_tstep*time[step_count] < sign_tstep*tspan[2]
            d.error_code = LIMIT_EXCEEDED
        end
        should_continue = false
    end
    should_continue &= sign_tstep*d.step_result.time < sign_tstep*tspan[2]
    should_continue &= !iszero(predicted_hj)
    should_continue &= (status_flag == RELAXATION_NOT_CALLED)
    should_continue
end

function display_iteration_summary(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M,T,S,F,K,X,NY}
    #println("Step = #$(d.step_count), Storage = $(d.storage[d.step_count]), Storage Apriori = $(d.storage_apriori[d.step_count])")
end

function DBB.relax!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M,T,S,F,K,X,NY}
    initialize_relax!(d)               # Reset storage used when relaxations are computed and times
    tstart = time()
    compute_X0!(d)                     # Compute initial condition values
    while continue_relax_loop!(d)
        display_iteration_summary(d)   # Display a single step results
        single_step!(d)                # Perform a single step
        store_step_result!(d)          # Store results from a single step
    end
    display_iteration_summary(d)       # Display a single step results
    clean_results!(d)                  # Resize storage to computed values and set completion code (if unset)
    if d.print_relax_time 
        println("relax time = $(time() - tstart)")
    end
end
