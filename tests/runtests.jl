#!/usr/bin/env julia
using Test, DynamicBoundspODEsPILMS #, DataStructures

@testset "Discretize and Relax" begin

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
    #DynamicBoundspODEsPILMS.compute_coefficients!(plms)
    #plms.β[1] = 1.0
    #plms.β[2] = 3.0

    #DynamicBoundspODEsPILMS.eval_cycle_Jx!(pf)
    #DynamicBoundspODEsPILMS.eval_cycle_Jp!(pf)
    #DynamicBoundspODEsPILMS.compute_sum_Jp!(pf)
    #DynamicBoundspODEsPILMS.compute_δₖ!(pf)
end
