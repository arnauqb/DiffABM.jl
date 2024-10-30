using NNlib

neighbours = [[1, 3], [2, 5, 6], [8, 9]]
x = [10, 20, 30, 40, 50, 60, 70, 80, 90]

map(i -> Base.getindex(x, i), neighbours)

##
using Zygote
function testdict(x)
    dict = Dict(i => 2.0 for i in 1:length(x))
    dict[2] = dict[2] * x[1]^2
    return sum(values(dict))
end

x = rand(10)
Zygote.gradient(x -> testdict(x), x)


##

