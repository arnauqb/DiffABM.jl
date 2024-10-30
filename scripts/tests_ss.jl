using PyPlot
"""
    generate_vision_matrices(max_vision)

Returns a matrix of size (2 * max_vision+1, 2 * max_vision+1) with zeros everywhere,
except for the squares that are away from the center by a distance less than or equal to vision_radius.
vision_radius varies from 1 to max_vision.
"""
function generate_vision_matrices(max_vision)
    vision_matrices = []
    N = 2 * max_vision + 1
    for vision_radius in 1:max_vision
        vision_matrix = zeros(N, N)
        for i in 1:N
            for j in 1:N
                if abs(i - max_vision - 1) + abs(j - max_vision - 1) <= vision_radius
                    vision_matrix[i, j] = 1
                end
            end
        end
        push!(vision_matrices, vision_matrix)
    end
    return vision_matrices
end

vision_matrices = generate_vision_matrices(3)
fig, ax = plt.subplots(1, length(vision_matrices))
for i in 1:3
    ax[i].imshow(vision_matrices[i])
end
fig

##
weights = [1.0, 0.6, 0.3]
weighted_matrices = [vision_matrices[i] * weights[i] for i in 1:3]
weighted_matrices_cumsum = cumsum(weighted_matrices)
fig, ax = plt.subplots()
for i in 1:3
    ax.imshow(weighted_matrices_cumsum[i], cmap="Blues")
end
color_indices = weights
colors = plt.cm.Blues(color_indices)
# make a legend showing a box for each color_indices
legend_elements = [plt.Line2D([0], [0], color=colors[i,:], lw=4) for i in 1:3]
ax.legend(legend_elements, ["$i" for i in 1:3], title="Vision radius")
fig
