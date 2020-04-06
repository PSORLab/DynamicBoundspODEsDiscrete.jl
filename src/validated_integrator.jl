"""
$(TYPEDSIGNATURES)
"""
compute_γ() = 1.0


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
function single_step!(f, hj_in, γ, hmin, P, Yⱼ)

    hj = hj_in
    hj1 = 0.0
    # validate existence & uniqueness
    hⱼ, Ỹⱼ, f̃, step_flag = existence_uniqueness(tf!, Yⱼ, P, hⱼ, hmin, f, ∂f∂y_in)
    if ~success_flag
        return success_flag
    end
    while hj > hmin
        parametric_lohners!(stf!, rtf!, dtf!, hⱼ, Ỹⱼ, Yⱼ, yⱼ, P, p,
                            Aⱼ₊₁, Aⱼ, Δⱼ, result, tjac, cfg, Jxsto, Jpsto, Jx, Jp)

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
    end
    return success_flag
end

function integrate!(γ)
    integrate_flag &= single_step!()
    step_size_error(integrate_flag)
end
