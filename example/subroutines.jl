using DynamicBoundspODEsPILMS, LinearAlgebra, IntervalArithmetic, StaticArrays, TaylorSeries,
      TaylorIntegration, ForwardDiff, McCormick, BenchmarkTools, DocStringExtensions

using DiffResults: JacobianResult, MutableDiffResult
using ForwardDiff: Partials, JacobianConfig, vector_mode_dual_eval, value, vector_mode_jacobian!

import Base.copyto!

function f!(dx,x,p,t)
    dx[1] = x[1]
    dx[2] = x[2]
    nothing
end
np = 2
nx = 2
k = 3
x = [1.0; 2.0]
p = [2.2; 2.2]
y = Interval{Float64}.([x; p])
g = DynamicBoundspODEsPILMS.JacTaylorFunctor!(f!, nx, np, k, Interval{Float64}(0.0), 0.0)
out = g.out
cfg = ForwardDiff.JacobianConfig(nothing, out, y)
result = JacobianResult(out, y)
xIntv = Interval{Float64}.(x)
pIntv = Interval{Float64}.(p)
tcoeffs = DynamicBoundspODEsPILMS.jacobian_taylor_coeffs!(result, g, xIntv, pIntv, cfg)
#@btime jacobian_taylor_coeffs!($result, $g, $xIntv, $pIntv, $cfg)
jac = result.derivs[1]
tjac = zeros(Interval{Float64}, 4, 8)
val = result.value

Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
DynamicBoundspODEsPILMS.extract_JxJp!(Jx, Jp, result, tjac, nx, np, k)
#@btime extract_JxJp!($Jx, $Jp, $result, $tjac, $nx, $np, $k)

# extact is good... actual jacobians look odd...

hⱼ = 0.001
hmin = 0.00001
Yⱼ = [Interval(0.1, 5.1); Interval(0.1, 8.9)]
P = [Interval(0.1, 5.1); Interval(0.1, 8.9)]
routIntv = copy(Yⱼ)
DynamicBoundspODEsPILMS.existence_uniqueness(g, Yⱼ, P, hⱼ, hmin, routIntv, Jx)
#@btime improvement_condition($Yⱼ, $Yⱼ, $nx)
@btime DynamicBoundspODEsPILMS.existence_uniqueness($g, $Yⱼ, $P, $hⱼ, $hmin, $routIntv, $Jx)
#tv, xv = validated_integration(f!, Interval{Float64}.([3.0, 3.0]), 0.0, 0.3, 4, 1.0e-20, maxsteps=100 )
Q = [Yⱼ; P]
#@btime jacobianfunctor($outIntv, $yInterval)

d = jacobianfunctor
zqwa = d.g!
zqwb = d.t
zqwc = d.xtaylor
zqwd = d.xout
zqwe = d.xaux
zqwr = d.taux
#@btime jetcoeffs!($zqwa, $zqwb, $zqwc, $zqwd, $zqwe, $zqwr, $s, $p)
#@code_warntype jetcoeffs!(zqwa, zqwb, zqwc, zqwd, zqwe, zqwr, p)


Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jxsto = zeros(Interval{Float64},2,2)
Jpsto = zeros(Interval{Float64},2,2)

rtf  = DynamicBoundspODEsPILMS.TaylorFunctor!(f!, nx, np, k, zero(Float64), zero(Float64))
Yⱼ = [Interval{Float64}(-10.0, 20.0); Interval{Float64}(-10.0, 20.0)]
P = [Interval{Float64}(2.0, 3.0); Interval{Float64}(2.0, 3.0)]
yⱼ = mid.(Yⱼ)
Δⱼ = Yⱼ - yⱼ
At = zeros(2,2) + I
Aⱼ =  DynamicBoundspODEsPILMS.QRDenseStorage(nx)
Aⱼ₊₁ =  DynamicBoundspODEsPILMS.QRDenseStorage(nx)
itf = DynamicBoundspODEsPILMS.TaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
dtf = g
hⱼ = 0.001
# TODO: Remember rP is computed outside iteration and stored to JacTaylorFunctor
plohners = DynamicBoundspODEsPILMS.parametric_lohners!(itf, rtf, dtf, hⱼ, Yⱼ, Yⱼ, yⱼ,
                                     P, p, Aⱼ₊₁, Aⱼ, Δⱼ, result, tjac, cfg,
                                     Jxsto, Jpsto, Jx, Jp)

@btime DynamicBoundspODEsPILMS.parametric_lohners!($itf, $rtf, $dtf, $hⱼ, $Yⱼ, $Yⱼ, $yⱼ,
                                 $P, $p, $Aⱼ₊₁, $Aⱼ, $Δⱼ, $result, $tjac, $cfg,
                                 $Jxsto, $Jpsto, $Jx, $Jp)
