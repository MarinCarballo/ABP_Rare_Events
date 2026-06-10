# Histogram and generic IO path helpers.

function abp_bin_index_from_edges(x::Real, edges::AbstractVector)
    i = searchsortedlast(edges, x)
    if i == length(edges) && x == last(edges)
        i -= 1
    end
    return (1 <= i < length(edges)) ? i : nothing
end

function abp_add_weighted_value!(counts::AbstractVector, edges::AbstractVector, value::Real, weight::Real)
    i = abp_bin_index_from_edges(value, edges)
    i === nothing && return false
    counts[i] += weight 
    return true
end #this function makes it easy to add weighted samples to a histogram defined by edges, returning false if the value is out of bounds.

function abp_add_weighted_joint!(counts::AbstractMatrix, x_edges::AbstractVector, y_edges::AbstractVector, x::Real, y::Real, weight::Real)
    ix = abp_bin_index_from_edges(x, x_edges)
    iy = abp_bin_index_from_edges(y, y_edges)
    (ix === nothing || iy === nothing) && return false
    counts[ix, iy] += weight
    return true
end

function abp_centers_from_edges(edges::AbstractVector)
    return 0.5 .* (edges[1:end-1] .+ edges[2:end])
end

function abp_pdf_from_mass(counts::AbstractVector, edges::AbstractVector)
    total = sum(counts)
    total == 0.0 && return fill(NaN, length(counts))
    return counts ./ (total .* diff(edges))
end

function abp_pdf2_from_mass(counts::AbstractMatrix, x_edges::AbstractVector, y_edges::AbstractVector)
    total = sum(counts)
    total == 0.0 && return fill(NaN, size(counts))
    dx = diff(x_edges)
    dy = diff(y_edges)
    return counts ./ (total .* (dx .* dy'))
end

function abp_unbias_weight_from_value(
    bias_value::Real,
    bias_edges::AbstractVector,
    logw_values::AbstractVector,
    logw_shift::Real,
)
    i = abp_bin_index_from_edges(bias_value, bias_edges)
    i === nothing && return 0.0
    lw = logw_values[i]
    isfinite(lw) || return 0.0
    return exp(-(lw - logw_shift))
end

function abp_in_endpoint_window(xT::Real, lo::Real, hi::Real)
    isinf(lo) && lo < 0 && isinf(hi) && hi > 0 && return true
    return lo <= xT < hi
end

function abp_run_tag(D::Real)
    Dtag = replace(string(D), "." => "p", "-" => "m")
    return "D$(Dtag)"
end

function abp_ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

