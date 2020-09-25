#!/usr/bin/env julia
using Test, DynamicBoundspODEsDiscrete, IntervalArithmetic
using DiffResults: JacobianResult

const DR = DynamicBoundspODEsDiscrete

@testset "Discretize and Relax" begin

    # test improvement condition for existence & uniqueness
    Yold = [Interval(1.0, 3.0); Interval(2.0, 4.0); Interval(1.0, 3.0)]
    Ynew = [Interval(1.5, 2.0); Interval(3.0, 3.5); Interval(0.5, 3.5)]
    nx_ic = 2
    @test DR.improvement_condition(Yold, Ynew, nx_ic)

    # construct storage for QR factorizations
    storage = DR.QRDenseStorage(nx_ic)
    @test storage.factorization.Q[1,1] == -1.0
    @test storage.factorization.Q[2,2] == 1.0
    @test storage.factorization.R[1,1] == -1.0
    @test storage.factorization.R[2,2] == 1.0

    DR.calculateQ!(storage, [1.0 3.0; 2.0 1.0], nx_ic)
    @test isapprox(storage.Q[1,1], -0.447214, atol = 1E-3)
    @test isapprox(storage.Q[1,2], -0.894427, atol = 1E-3)
    @test isapprox(storage.Q[2,1], -0.894427, atol = 1E-3)
    @test isapprox(storage.Q[2,2], 0.4472135, atol = 1E-3)

    # results in symmetric matrix
    DR.calculateQinv!(storage)
    @test storage.inv[1,1] == storage.Q[1,1]
    @test storage.inv[1,2] == storage.Q[1,2]
    @test storage.inv[2,1] == storage.Q[2,1]
    @test storage.inv[2,2] == storage.Q[2,2]

    stack = DR.qr_stack(nx_ic, 3)
    DR.reinitialize!(stack)
    @test stack[1].Q[1,1] == 1.0
    @test stack[1].Q[1,2] == 0.0
    @test stack[1].Q[2,1] == 0.0
    @test stack[1].Q[2,2] == 1.0

    # eval_cycle! test
    buffs = DR.CircularBuffer(zeros(2,2),3)

    function J!(out, x, p ,t)
        out[1,1] = x
        nothing
    end

    DR.eval_cycle!(J!, buffs, 1.0, 0.0, 0.0)
    DR.eval_cycle!(J!, buffs, 2.0, 0.0, 0.0)
    DR.eval_cycle!(J!, buffs, 3.0, 0.0, 0.0)
    @test buffs.buffer[1][1,1] == 3.0
    @test buffs.buffer[2][1,1] == 2.0
    @test buffs.buffer[3][1,1] == 1.0

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
    jtf! = DR.JacTaylorFunctor!(f!, nx, np, Val(k), Interval{Float64}(0.0), 0.0)

    xIntv = Interval{Float64}.(x)
    pIntv = Interval{Float64}.(p)
    yIntv = [xIntv; pIntv]
    DR.jacobian_taylor_coeffs!(jtf!, xIntv, pIntv, 0.0)

    jac = JacobianResult(jtf!.out, yIntv).derivs[1]
    tjac = zeros(Interval{Float64}, nx + np, nx*(k+1))
    Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},nx,nx) for i in 1:(k+1)]
    Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},nx,np) for i in 1:(k+1)]

    DR.set_JxJp!(jtf!, xIntv, pIntv, 0.0)
    @test isapprox(jtf!.Jp[2][2,1].lo, 0.4, atol=1E-3)
    @test isapprox(jtf!.Jp[2][1,2].lo, 1.0, atol=1E-3)
    @test isapprox(jtf!.Jp[4][2,1].lo, 0.0666666, atol=1E-3)
    @test isapprox(jtf!.Jp[4][1,2].lo, 0.079999, atol=1E-3)
    @test isapprox(jtf!.Jx[2][1,1].lo, 0.2, atol=1E-3)
    @test isapprox(jtf!.Jx[2][2,2].lo, 1.0, atol=1E-3)
    @test isapprox(jtf!.Jx[4][1,1].lo, 0.030666, atol=1E-3)
    @test isapprox(jtf!.Jx[4][2,2].lo, 0.1666, atol=1E-3)

    # make/evaluate interval valued Taylor cofficient functor
    itf! = DR.TaylorFunctor!(f!, nx, np, Val(k), zero(Interval{Float64}), zero(Float64))
    outIntv = Vector{Interval{Float64}}[zeros(Interval{Float64},2) for i in 1:4]
    itf!(outIntv, xIntv, pIntv, 0.0)
    @test isapprox(outIntv[1][1].lo, 0.10001, atol=1E-3)
    @test isapprox(outIntv[2][2].lo, 1.0399999999999998, atol=1E-3)
    @test isapprox(outIntv[3][1].lo, 0.011, atol=1E-3)
    @test isapprox(outIntv[4][2].lo, 0.173334, atol=1E-3)
    @test isapprox(outIntv[1][2].hi, 1.0, atol=1E-3)
    @test isapprox(outIntv[2][1].hi, 0.1100000000000000, atol=1E-3)
    @test isapprox(outIntv[3][2].hi, 0.52, atol=1E-3)
    @test isapprox(outIntv[4][1].hi, 0.004766666666666669, atol=1E-3)

    # make/evaluate real valued Taylor cofficient functor
    rtf!  = DR.TaylorFunctor!(f!, nx, np, Val(k), zero(Float64), zero(Float64))
    out = Vector{Float64}[zeros(Float64,2) for i in 1:4]
    rtf!(out, x, p, 0.0)
    @test isapprox(out[1][1], 0.10001, atol=1E-3)
    @test isapprox(out[2][1], 0.11000000000000001, atol=1E-3)
    @test isapprox(out[3][1], 0.011, atol=1E-3)
    @test isapprox(out[4][1], 0.004766666666666668, atol=1E-3)
    @test isapprox(out[1][2], 1.0, atol=1E-3)
    @test isapprox(out[2][2], 1.04, atol=1E-3)
    @test isapprox(out[3][2], 0.52, atol=1E-3)
    @test isapprox(out[4][2], 0.17333333333333334, atol=1E-3)

    # higher order existence tests
    hⱼ = 0.001
    hmin = 0.00001
    function euf!(out, x, p, t)
        out[1,1] = -x[1]^2
        nothing
    end

    jtf_exist_unique! = DR.JacTaylorFunctor!(euf!, 1, 1, Val(k), Interval{Float64}(0.0), 0.0)
    xIntv_plus =  xIntv .+ Interval(0,1)
    DR.jacobian_taylor_coeffs!(jtf_exist_unique!, xIntv_plus, pIntv, 0.0)
    @test isapprox(jtf_exist_unique!.result.value[4].lo, -1.464100000000001, atol=1E-5)
    @test isapprox(jtf_exist_unique!.result.value[4].hi, -9.999999999999999e-5, atol=1E-5)
    @test isapprox(jtf_exist_unique!.result.derivs[1][3,1].lo, 0.03, atol=1E-5)
    @test isapprox(jtf_exist_unique!.result.derivs[1][3,1].hi, 3.6300000000000012, atol=1E-5)

    coeff_out = zeros(Interval{Float64},1,k)
    DR.set_JxJp!(jtf_exist_unique!, xIntv_plus, pIntv, 0.0)
    @test isapprox(jtf_exist_unique!.Jx[4][1,1].lo, -5.32401, atol=1E-5)
    @test isapprox(jtf_exist_unique!.Jx[4][1,1].hi, -0.00399999, atol=1E-5)
    @test jtf_exist_unique!.Jp[1][1,1].lo == jtf_exist_unique!.Jp[1][1,1].hi == 0.0

    #u_result = DR.UniquenessResult(1,1)
    #DR.existence_uniqueness!(u_result, itf_exist_unique!, eufY, hⱼ, hmin,
    #                         coeff_out_exist_unique!, Jx, Jp)

    #@test u_result.step == 0.001
    #@test u_result.confirmed
end
