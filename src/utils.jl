function wrap_index(N, i, j)
    # given board length N, and index i, j
    # return the wrapped index to make the board toroidal
    return mod1(i, N), mod1(j, N)
end