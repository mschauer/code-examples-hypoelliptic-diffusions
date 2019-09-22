workdir = @__DIR__
cd(workdir)

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using RCall

T = 0.5
dt = 1/5000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

obs_scheme =["full","firstcomponent"][2]  # either condition on all components, or only first component
easy_conditioning = false # if true, then path to 1, else to 2

Σdiagel = 10^(-10)

# settings sampler
iterations = 1_000
skip_it = 1
subsamples = 0:skip_it:iterations

ρ = obs_scheme=="full" ? 0.85 : 0.95

ρ = 0.98

if obs_scheme=="full"
    L = SMatrix{3,3}(1.0I)
    v = easy_conditioning ?  ℝ{3}(1/32,1/4,1) :  ℝ{3}(5/128,3/8,2)
end
if obs_scheme=="firstcomponent"
    L = @SMatrix [1. 0. 0.]
    v = easy_conditioning ? 1/32 : 5/128
end

m, d = size(L)
Σ = SMatrix{m,m}(Σdiagel*I)

# specify target process
struct NclarDiffusion <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.b(t, x, P::NclarDiffusion) = ℝ{3}(x[2],x[3],-P.α * sin(P.ω * x[3]))
Bridge.σ(t, x, P::NclarDiffusion) = ℝ{3}(0.0, 0.0, P.σ)
Bridge.constdiff(::NclarDiffusion) = true

P = NclarDiffusion(2*3.0, 2pi, 1.0)
x0 = ℝ{3}(0.0, 0.0, 0.0)

# specify auxiliary process
struct NclarDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Random.seed!(4)
Bridge.B(t, P::NclarDiffusionAux) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 0.0]
Bridge.β(t, P::NclarDiffusionAux) = ℝ{3}(0.0,0.0,0)
Bridge.σ(t,  P::NclarDiffusionAux) = ℝ{3}(0.0,0.0, P.σ)
Bridge.constdiff(::NclarDiffusionAux) = true
Bridge.b(t, x, P::NclarDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::NclarDiffusionAux) = Bridge.σ(t,P) * Bridge.σ(t,  P)'

Pt = NclarDiffusionAux(P.α, P.ω, P.σ)

# Solve Backward Recursion
Po = Bridge.PartialBridge(tt, P, Pt, L, ℝ{m}(v), Σ)

####################### MH algorithm ###################
# initalisation
W = sample(tt, Wiener())
X = solve(Euler(), x0, W, P) # to make the object
Xo = copy(X)
solve!(Euler(),X, x0, W, Po)
ll = llikelihood(Bridge.LeftRule(), X, Po,skip=sk)
Xo = copy(X)
Wo = copy(W)
W2 = copy(W)
Wo = copy(W)
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

iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:3, j in 1:10:length(X), (i,s) in enumerate(subsamples) ][:]
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
            fct_recode(component,'component 1'='1','component 2'='2','component 3'='3')) %>%
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
