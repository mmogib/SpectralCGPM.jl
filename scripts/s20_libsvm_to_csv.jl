using CSV
using DataFrames

const JCODE_ROOT = normpath(joinpath(@__DIR__, ".."))
const LIBSVM_DIR = joinpath(JCODE_ROOT, "data", "libsvm")

const FEATURE_COUNTS = Dict(
    "colon-cancer" => 2000,
)

function inferred_feature_count(name::AbstractString, observed_max::Int)
    if haskey(FEATURE_COUNTS, name)
        return FEATURE_COUNTS[name]
    elseif startswith(name, "a")
        return max(observed_max, 123)
    elseif startswith(name, "w")
        return max(observed_max, 300)
    end
    return observed_max
end

function parse_libsvm(file_path::AbstractString)
    labels = Float64[]
    rows = Vector{Dict{Int, Float64}}()
    observed_max = 0

    open(file_path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue

            parts = split(line)
            push!(labels, parse(Float64, parts[1]))

            features = Dict{Int, Float64}()
            for token in Iterators.drop(parts, 1)
                idx_text, value_text = split(token, ':', limit = 2)
                idx = parse(Int, idx_text)
                value = parse(Float64, value_text)
                features[idx] = value
                observed_max = max(observed_max, idx)
            end
            push!(rows, features)
        end
    end

    return labels, rows, observed_max
end

function to_dataframe(labels::Vector{Float64}, rows::Vector{Dict{Int, Float64}}, feature_count::Int)
    cols = [Vector{Float64}(undef, length(labels)) for _ in 1:feature_count]

    for (row_idx, features) in pairs(rows)
        for feature_idx in 1:feature_count
            cols[feature_idx][row_idx] = get(features, feature_idx, 0.0)
        end
    end

    df = DataFrame(label = labels)
    for feature_idx in 1:feature_count
        df[!, Symbol("x$(feature_idx)")] = cols[feature_idx]
    end
    return df
end

function convert_file(input_path::AbstractString)
    name = basename(input_path)
    labels, rows, observed_max = parse_libsvm(input_path)
    feature_count = inferred_feature_count(name, observed_max)
    df = to_dataframe(labels, rows, feature_count)
    output_path = joinpath(dirname(input_path), "$(name).csv")
    CSV.write(output_path, df)
    println("Wrote $(basename(output_path)) with $(nrow(df)) rows and $(ncol(df) - 1) feature columns")
end

function main()
    files = sort(filter(name -> !endswith(name, ".csv"), readdir(LIBSVM_DIR)))
    for name in files
        convert_file(joinpath(LIBSVM_DIR, name))
    end
end

main()
