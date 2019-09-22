workdir = @__DIR__
cd(workdir)

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using RCall

T = 12.0 #2.0
dt = 1/5000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

# settings in case of νH - parametrisation
Σdiagel = 10^(-10)

iterations = 1_000 # 50_000
skip_it = 1     # 1000
subsamples = 0:skip_it:iterations

# specify observation scheme
L = @SMatrix [1. 0.]
m, d = size(L)

Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]

# specify target process
struct FitzhughDiffusion <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
end

Bridge.b(t, x, P::FitzhughDiffusion) = ℝ{2}((x[1]-x[2]-x[1]^3+P.s)/P.ϵ, P.γ*x[1]-x[2] +P.β)
Bridge.σ(t, x, P::FitzhughDiffusion) = ℝ{2}(0.0, P.σ)
Bridge.constdiff(::FitzhughDiffusion) = true

P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3) # settings of Ditlevsen-Samson
x0 = ℝ{2}(-0.5, -0.6)

# specify auxiliary process
struct FitzhughDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
    t::Float64
    u::Float64
    T::Float64
    v::Float64
end

function uv(t, P::FitzhughDiffusionAux)
    λ = (t - P.t)/(P.T - P.t)
    P.v*λ + P.u*(1-λ)
end

# specify type of auxiliary process
k1 = k2 = 1
aux_choice = ["linearised_end" "linearised_startend"  "matching"][k1]
endpoint = ["first", "extreme"][k2]


Random.seed!(44)

if endpoint == "first"
    v = -1
elseif endpoint == "extreme"
    v = 1.1
else
    error("not implemented")
end

if aux_choice=="linearised_end"
    Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*P.v^2/P.ϵ  -1/P.ϵ; P.γ -1.0]
    Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ+2*P.v^3/P.ϵ, P.β)
    ρ = endpoint=="extreme" ? 0.9 : 0.0
elseif aux_choice=="linearised_startend"
    Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*uv(t, P)^2/P.ϵ  -1/P.ϵ; P.γ -1.0]
    Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ+2*uv(t, P)^3/P.ϵ, P.β)
    ρ = endpoint=="extreme" ? 0.98 : 0.0
else
    Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ  -1/P.ϵ; P.γ -1.0]
    Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ-(P.v^3)/P.ϵ, P.β)
    ρ = 0.99
end

Bridge.σ(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ)
Bridge.constdiff(::FitzhughDiffusionAux) = true

Bridge.b(t, x, P::FitzhughDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::FitzhughDiffusionAux) = Bridge.σ(t,P) * Bridge.σ(t, P)'

Pt = FitzhughDiffusionAux(P.ϵ, P.s, P.γ, P.β, P.σ, tt[1], x0[1], tt[end], v)

# Solve Backward Recursion
Po = Bridge.PartialBridge(tt, P, Pt, L, ℝ{m}(v), Σ)

####################### MH algorithm ###################
# initalisation
W = sample(tt, Wiener())
X = solve(Euler(), x0, W, P) # to make the object
solve!(Euler(),X, x0, W, Po)
ll = llikelihood(Bridge.LeftRule(), X, Po,skip=sk)
Xo = copy(X)
Wo = copy(W)
W2 = copy(W)

XX = typeof(X)[]
if 0 in subsamples
    push!(XX, copy(X))
end

acc = 0

for iter in 1:iterations
    global ll
    global acc
    sample!(W2, Wiener())
    Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
    solve!(Euler(),Xo, x0, Wo, Po)
    llo = llikelihood(Bridge.LeftRule(), Xo, Po)
    if log(rand()) <= llo - ll
        X.yy .= Xo.yy
        W.yy .= Wo.yy
        println("iter ", iter," diff loglik ", round(llo-ll; digits=3), "✓")
        ll = llo
        acc +=1
    end
    if iter in subsamples
        push!(XX, copy(X))
        println("iter ", iter," diff loglik ", round(llo-ll; digits=3), "")
    end
end
@info "Done."*"\x7"^6


ave_acc_perc = 100*round(acc/iterations; digits=2)

# write info to txt file
fn = workdir*"/info.txt"
f = open(fn,"w")
write(f, "Choice of auxiliary process: ",aux_choice,"\n")
write(f, "Choice of endpoint: ",endpoint,"\n\n")
write(f, "Number of iterations: ",string(iterations),"\n")
write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
write(f, "Starting point: ",string(x0),"\n")
write(f, "End time T: ", string(T),"\n")
write(f, "Endpoint v: ",string(v),"\n")
write(f, "Noise Sigma: ",string(Σ),"\n")
write(f, "L: ",string(L),"\n\n")
write(f,"Mesh width: ",string(dt),"\n")
write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n")
close(f)


println("Average acceptance percentage: ",ave_acc_perc,"\n")

extractcomp(d,k) = map(x->x[k],d)

iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:10:length(X), (i,s) in enumerate(subsamples) ][:]
d = DataFrame(iteration = extractcomp(iterates,1), time = extractcomp(iterates,2),
        component = extractcomp(iterates,3), value = extractcomp(iterates,4))

skip_infig = 100 # skip every skip_infig iteration in the figure

@rput skip_infig
@rput d
R"""
library(ggplot2)
library(ggthemes)
library(tidyverse)
library(gridExtra)

d$component <- as.factor(d$component)
d <- d %>% mutate(component=
            fct_recode(component,'component 1'='1','component 2'='2')) %>%
            filter(iteration %in% seq(0,max(d$iteration),by=skip_infig))

p <- ggplot(mapping=aes(x=time,y=value,colour=iteration),data=d) +
    geom_path(aes(group=iteration)) +
    facet_wrap(~component,ncol=1,scales='free_y')+
    scale_colour_gradient(low='green',high='blue')+
    ylab("")

# pdf("paths.pdf",width=7,height=4)
  show(p)
# dev.off()

"""
