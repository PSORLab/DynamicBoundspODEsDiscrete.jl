#!/usr/bin/env julia
using Test, DynamicBoundspODEsPILMS, IntervalArithmetic
using DiffResults: JacobianResult

@testset "Discretize and Relax" begin

    # test improvement condition for existence & uniqueness
    Yold = [Interval(1.0, 3.0); Interval(2.0, 4.0); Interval(1.0, 3.0)]
    Ynew = [Interval(1.5, 2.0); Interval(3.0, 3.5); Interval(0.5, 3.5)]
    nx_ic = 2
    @test DynamicBoundspODEsPILMS.improvement_condition(Yold, Ynew, nx_ic)

    # construct storage for QR factorizations
    storage = DynamicBoundspODEsPILMS.QRDenseStorage(nx_ic)
    @test storage.factorization.Q[1,1] == -1.0
    @test storage.factorization.Q[2,2] == 1.0
    @test storage.factorization.R[1,1] == -1.0
    @test storage.factorization.R[2,2] == 1.0

    DynamicBoundspODEsPILMS.calculateQ!(storage, [1.0 3.0; 2.0 1.0], nx_ic)
    @test isapprox(storage.Q[1,1], -0.447214, atol = 1E-3)
    @test isapprox(storage.Q[1,2], -0.894427, atol = 1E-3)
    @test isapprox(storage.Q[2,1], -0.894427, atol = 1E-3)
    @test isapprox(storage.Q[2,2], 0.4472135, atol = 1E-3)

    # results in symmetric matrix
    DynamicBoundspODEsPILMS.calculateQinv!(storage)
    @test storage.inv[1,1] == storage.Q[1,1]
    @test storage.inv[1,2] == storage.Q[1,2]
    @test storage.inv[2,1] == storage.Q[2,1]
    @test storage.inv[2,2] == storage.Q[2,2]

    stack = DynamicBoundspODEsPILMS.qr_stack(nx_ic, 3)
    DynamicBoundspODEsPILMS.reinitialize!(stack)
    @test stack[1].Q[1,1] == 1.0
    @test stack[1].Q[1,2] == 0.0
    @test stack[1].Q[2,1] == 0.0
    @test stack[1].Q[2,2] == 1.0

    # eval_cycle! test
    buffs = DynamicBoundspODEsPILMS.CircularBuffer(zeros(2,2),3)

    function J!(out, x, p ,t)
        out[1,1] = x
        nothing
    end

    DynamicBoundspODEsPILMS.eval_cycle!(J!, buffs, 1.0, 0.0, 0.0)
    DynamicBoundspODEsPILMS.eval_cycle!(J!, buffs, 2.0, 0.0, 0.0)
    DynamicBoundspODEsPILMS.eval_cycle!(J!, buffs, 3.0, 0.0, 0.0)
    @test buffs.buffer[1][1,1] == 3.0
    @test buffs.buffer[2][1,1] == 2.0
    @test buffs.buffer[3][1,1] == 1.0

    plms = DynamicBoundspODEsPILMS.PLMS(Val(2), AdamsMoulton())
    append!(plms.times, [3.0; 1.5; 1.0])
    #compute_coefficients!(plms)
    #println(plms.coeffs)
    #@test plms.coeffs[1] ==
    #@test plms.coeffs[2] ==

    function Jx!(out::Matrix{S}, x, p ,t) where S
        fill!(out, zero(S))
        out[1,1] = x[1]
        out[2,2] = x[2]
        nothing
    end
    function Jp!(out::Matrix{S}, x, p ,t) where S
        fill!(out, zero(S))
        out[2,1] = p[1]
        out[1,2] = p[2]
        nothing
    end
    nx = 2
    np = 2
    plms_functor = DynamicBoundspODEsPILMS.PLMsFunctor(zero(Interval{Float64}),
                                                       plms, Jx!, Jp!, nx, np)
    #DynamicBoundspODEsPILMS.compute_coefficients!(plms)
    #plms.β[1] = 1.0
    #plms.β[2] = 3.0

    #DynamicBoundspODEsPILMS.eval_cycle_Jx!(pf)
    #DynamicBoundspODEsPILMS.eval_cycle_Jp!(pf)
    #DynamicBoundspODEsPILMS.compute_sum_Jp!(pf)
    #DynamicBoundspODEsPILMS.compute_δₖ!(pf)

    out = zeros(2,2)
    a = [1.0; 2.0; 3.0; 4.0]
    DynamicBoundspODEsPILMS.coeff_to_matrix!(out, a, 2, 1)
    @test out[1,1] == 1.0
    @test out[2,1] == 2.0
    @test out[1,2] == 3.0
    @test out[2,2] == 4.0

    function f!(dx,x,p,t)
        dx[1] = x[1]^2 + p[2]
        dx[2] = x[2] + p[1]^2
        nothing
    end
    np = 2
    nx = 2
    k = 3
    x = [0.1; 1.0]
    p = [0.2; 0.1]
    jtf! = DynamicBoundspODEsPILMS.JacTaylorFunctor!(f!, nx, np, k, Interval{Float64}(0.0), 0.0)

    xIntv = Interval{Float64}.(x)
    pIntv = Interval{Float64}.(p)
    yIntv = [xIntv; pIntv]
    DynamicBoundspODEsPILMS.jacobian_taylor_coeffs!(jtf!, yIntv)

    jac = JacobianResult(jtf!.out, yIntv).derivs[1]
    tjac = zeros(Interval{Float64}, nx+np, nx*(k+1))
    Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},nx,nx) for i in 1:(k+1)]
    Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},nx,np) for i in 1:(k+1)]

    DynamicBoundspODEsPILMS.extract_JxJp!(Jx, Jp, jtf!.result, tjac, nx, np, k)
    @test isapprox(Jp[2][2,1].lo, 0.4, atol=1E-3)
    @test isapprox(Jp[2][1,2].lo, 1.0, atol=1E-3)
    @test isapprox(Jp[4][2,1].lo, 0.0666666, atol=1E-3)
    @test isapprox(Jp[4][1,2].lo, 0.079999, atol=1E-3)
    @test isapprox(Jx[2][1,1].lo, 0.2, atol=1E-3)
    @test isapprox(Jx[2][2,2].lo, 1.0, atol=1E-3)
    @test isapprox(Jx[4][1,1].lo, 0.030666, atol=1E-3)
    @test isapprox(Jx[4][2,2].lo, 0.1666, atol=1E-3)

    coeff_out = zeros(Interval{Float64},2,4)
    DynamicBoundspODEsPILMS.coeff_to_matrix!(coeff_out, jtf!.out, nx, k)
    @test isapprox(coeff_out[1,1].hi, 0.100001, atol=1E-3)
    @test isapprox(coeff_out[2,2].hi, 1.039999, atol=1E-3)
    @test isapprox(coeff_out[1,4].hi, 0.0047666, atol=1E-3)
    @test isapprox(coeff_out[2,4].hi, 0.173333, atol=1E-3)

    # make/evaluate interval valued Taylor cofficient functor
    itf! = DynamicBoundspODEsPILMS.TaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
    outIntv = zeros(Interval{Float64},8)
    itf!(outIntv, yIntv)
    @test isapprox(outIntv[1].hi, 0.10001, atol=1E-3)
    @test isapprox(outIntv[3].hi, 0.011001, atol=1E-3)
    @test isapprox(outIntv[6].hi, 1.04001, atol=1E-3)
    @test isapprox(outIntv[8].hi, 0.173334, atol=1E-3)

    # make/evaluate real valued Taylor cofficient functor
    y = [x; p]
    rtf!  = DynamicBoundspODEsPILMS.TaylorFunctor!(f!, nx, np, k, zero(Float64), zero(Float64))
    out = zeros(8)
    rtf!(out, y)
    @test isapprox(outIntv[1], 0.10001, atol=1E-3)
    @test isapprox(outIntv[3], 0.011001, atol=1E-3)
    @test isapprox(outIntv[6], 1.04001, atol=1E-3)
    @test isapprox(outIntv[8], 0.173334, atol=1E-3)
end
