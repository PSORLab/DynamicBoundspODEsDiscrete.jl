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

function DBB.relax!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY}

    fill!(d.time, 0.0)

    storage_len = length(d.storage)
    for i=1:(1000-storage_len)
        push!(d.storage, zeros(T, d.nx))
        push!(d.storage_apriori, zeros(T, d.nx))
        push!(d.time, 0.0)
    end

    # reset relax!
    for i = 1:length(d.storage)
        fill!(d.storage[i], zero(T))
    end
    for i = 1:length(d.storage_apriori)
        fill!(d.storage_apriori[i], zero(T))
    end
    empty!(d.relax_t_dict_indx)
    empty!(d.relax_t_dict_flt)
    d.step_result.time = 0.0

    set_P!(d)         # Functor set P and P - p values for calculations
    compute_X0!(d)     # Compute initial condition values

    # Get initial time and integration direction
    t = d.tspan[1]
    tmax = d.tspan[2]
    sign_tstep = copysign(1, tmax - t)
    d.time[1] = t
    d.contractor_result.times[1] = t

    # Computes maximum step size to take (either hit support or max time)
    support_indx = 1
    next_support = Inf
    tsupports = d.tsupports
    if !isempty(tsupports)
        if (tsupports[1] == 0.0)
            support_indx += 1
        end
        next_support = tsupports[support_indx]
    end

    # initialize QR type storage
    set_Δ!(d.contractor_result.Δ, d.storage)
    fill!(d.contractor_result.A_Q[1], 0.0)
    fill!(d.contractor_result.A_inv[1], 0.0)
    for i = 1:d.nx
        d.contractor_result.A_Q[1][i,i] = 1.0
        d.contractor_result.A_inv[1][i,i] = 1.0
    end

    d.exist_result.status_flag = RELAXATION_NOT_CALLED

    # Begin integration loop
    hlast = 0.0
    d.step_count = 0
    is_adaptive = d.exist_result.hj <= 0.0
    d.exist_result.hj = !is_adaptive ? d.exist_result.hj : 0.01*(tmax - d.contractor_result.times[1])
    d.exist_result.predicted_hj = d.exist_result.hj
    stored_value_count = length(d.relax_t_dict_indx)
    for step_number = 2:(d.step_limit+2)
        if (sign_tstep*d.step_result.time <= sign_tstep*tmax) &&
           (d.exist_result.predicted_hj != 0.0)

            delT = abs(sign_tstep*d.step_result.time - sign_tstep*tmax)

            # max step size is min of predicted, when next support point occurs,
            # or the last time step in the span
            tv = d.step_result.time
            d.exist_result.hj = min(d.exist_result.hj, next_support - tv, tmax - tv)
            hj_limit = min(next_support - tv, tmax - tv)
            d.exist_result.hj_max = tmax - tv
            is_support_pnt = (hj_limit == next_support - tv)

            d.contractor_result.steps[1] = d.exist_result.hj
            d.contractor_result.step_count = step_number

            # perform step size calculation and update bound information
            single_step!(d.exist_result, d.contractor_result, d.step_params,
                         d.step_result, d.method_f!, step_number-1, hj_limit,
                         delT, d.constant_state_bounds, d.polyhedral_constraint)

            # unpack storage
            if step_number - 1 > length(d.time)
                push!(d.storage, copy(d.contractor_result.X_computed))
                push!(d.storage_apriori, copy(d.exist_result.Xj_apriori))
                push!(d.time, d.contractor_result.times[1])
            else
                copyto!(d.storage[step_number], d.contractor_result.X_computed)
                copyto!(d.storage_apriori[step_number], d.exist_result.Xj_apriori)
                d.time[step_number] = d.step_result.time
                d.contractor_result.times[1] = d.step_result.time
            end
            if is_support_pnt
                stored_value_count += 1
                d.relax_t_dict_indx[stored_value_count] = step_number
                d.relax_t_dict_flt[next_support] = step_number
                support_indx += 1
                if support_indx <= length(tsupports)
                    next_support = tsupports[support_indx]
                else
                    next_support = Inf
                end
            end

            # throw error if limit exceeded
            if d.exist_result.status_flag !== RELAXATION_NOT_CALLED
                d.error_code = d.exist_result.status_flag
                break
            end
            d.step_count += 1
        else
            break
        end
    end

    if (d.step_count > d.step_limit) && (sign_tstep*d.time[d.step_count + 1] < sign_tstep*tmax)
        d.error_code = LIMIT_EXCEEDED
    end

    # cut out any unnecessary array elements
    resize!(d.storage, d.step_count + 1)
    resize!(d.storage_apriori, d.step_count + 1)
    resize!(d.time, d.step_count + 1)

    if d.error_code === RELAXATION_NOT_CALLED
        d.error_code = COMPLETED
    end

    return nothing
end
