using Makie
p4 = Scene()


lines!(p4, Bridge.piecewise(SamplePath([vrange[1]-1; vrange; vrange[end]+1], counts))..., color=:darkorange, linewidth=3)

lines!(p4, Bridge.piecewise(SamplePath(vrange, wcounts))..., color=:darkblue, linestyle=:dash, linewidth=3.)
xlabel!(p4, "v")
ylabel!(p4, "p(v)")


bias =  wcounts - counts[2:end-1]
tb, upper = Bridge.piecewise(SamplePath(vrange, max.(0, bias)))
tb, lower = Bridge.piecewise(SamplePath(vrange, min.(0, bias)))
band!(p4, tb, lower, upper)
save(simname, p4)
p4
