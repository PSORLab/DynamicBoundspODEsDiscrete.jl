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
            next_support = tsupports[2]
            support_indx += 1
        else
            next_support = tsupports[1]
        end
    end

    # initialize QR type storage
    set_Δ!(d.contractor_result.Δ, d.storage)
    reinitialize!(d.contractor_result.A)

    # Begin integration loop
    hlast = 0.0
    d.step_count = 0
    is_adaptive = d.exist_result.hj <= 0.0
    d.exist_result.hj = !is_adaptive ? d.exist_result.hj : 0.01*(tmax - d.contractor_result.times[1])
    d.exist_result.predicted_hj = d.exist_result.hj

    for step_number = 2:(d.step_limit+2)
        if sign_tstep*d.step_result.time <= sign_tstep*tmax

            # max step size is min of predicted, when next support point occurs,
            # or the last time step in the span
            tv = d.step_result.time
            d.exist_result.hj = min(d.exist_result.hj, next_support - tv, tmax - tv)
            d.exist_result.hj_max = tmax - tv

            d.contractor_result.steps[1] = d.exist_result.hj
            d.contractor_result.step_count = step_number

            # perform step size calculation and update bound information
            single_step!(d.exist_result, d.contractor_result, d.step_params,
                         d.step_result, d.method_f!, step_number-1)

            # unpack storage
            if step_number - 1 > length(d.time)
                push!(d.storage, copy(d.contractor_result.X_computed))
                push!(d.storage_apriori, copy(d.exist_result.Xapriori))
                push!(d.time, d.contractor_result.times[1])
            end
            copy!(d.storage[step_number], d.contractor_result.X_computed)
            copy!(d.storage_apriori[step_number], d.exist_result.Xj_apriori)
            d.time[step_number] = d.step_result.time

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
    resize!(d.storage, d.step_count)
    resize!(d.storage_apriori, d.step_count)
    resize!(d.time, d.step_count)

    if d.error_code === RELAXATION_NOT_CALLED
        d.error_code = COMPLETED
    end

    return nothing
end
