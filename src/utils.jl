function wrap_index(N, i, j)
    # given board length N, and index i, j
    # return the wrapped index to make the board toroidal
    return mod(i - 1, N) + 1, mod(j - 1, N) + 1
end