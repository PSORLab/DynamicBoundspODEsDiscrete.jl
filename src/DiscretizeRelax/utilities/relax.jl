function DBB.relax!(d::DiscretizeRelax{M,T,S,F,K,X,NY}) where {M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY}

    set_P!(d) ::Nothing         # Functor set P and P - p values for calculations
    compute_X0!(d)::Nothing     # Compute initial condition values

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
    set_Δ!(d.contractor_result.Δ, d.storage)::Nothing
    reinitialize!(d.contractor_result.A)::Nothing

    # Begin integration loop
    hlast = 0.0
    step_number = 0
    d.step_count = 0
    is_adaptive = d.exist_result.hj <= 0.0
    d.exist_result.hj = !is_adaptive ? d.exist_result.hj : 0.01*(tmax - d.contractor_result.times[1])
    d.exist_result.predicted_hj = d.exist_result.hj

    while sign_tstep*d.time[step_number + 1] <= sign_tstep*tmax

        # max step size is min of predicted, when next support point occurs,
        # or the last time step in the span
        tv = d.time[step_number + 1]
        d.exist_result.hj = min(d.exist_result.hj, next_support - tv, tmax - tv)
        d.exist_result.hj_max = tmax - tv

        d.contractor_result.steps[1] = d.exist_result.hj
        d.contractor_result.step_count = step_number + 1

        # perform step size calculation and update bound information
        step_number = d.step_count + 1
        single_step!(d.exist_result, d.contractor_result, d.step_params,
                     d.step_result, d.method_f!, step_number)::Nothing

        # unpack storage
        if step_number > length(d.time)
            push!(d.storage, copy(d.contractor_result.X_computed))
            push!(d.storage_apriori, copy(d.exist_result.Xapriori))
            push!(d.time, d.contractor_result.times[1])
        end
        copy!(d.storage[step_number + 1], d.contractor_result.X_computed)
        copy!(d.storage_apriori[step_number + 1], d.exist_result.Xj_apriori)
        d.time[step_number + 1] = d.step_result.time

        # throw error if limit exceeded
        if step_number > d.step_limit
            d.error_code = LIMIT_EXCEEDED
            break
        elseif (d.exist_result.status_flag !== RELAXATION_NOT_CALLED)
            d.error_code = d.exist_result.status_flag
            break
        end
        d.step_count += 1
    end

    # cut out any unnecessary array elements
    resize!(d.storage, step_number)
    resize!(d.storage_apriori, step_number)
    resize!(d.time, step_number)

    if d.error_code === RELAXATION_NOT_CALLED
        d.error_code = COMPLETED
    end

    return nothing
end
