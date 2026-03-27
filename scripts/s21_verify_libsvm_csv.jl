const JCODE_ROOT = normpath(joinpath(@__DIR__, ".."))
const LIBSVM_DIR = joinpath(JCODE_ROOT, "data", "libsvm")

const EXPECTED_FEATURES = Dict(
    "colon-cancer" => 2000,
)

function inferred_feature_count(name::AbstractString, observed_max::Int)
    if haskey(EXPECTED_FEATURES, name)
        return EXPECTED_FEATURES[name]
    elseif startswith(name, "a")
        return max(observed_max, 123)
    elseif startswith(name, "w")
        return max(observed_max, 300)
    end
    return observed_max
end

function parse_libsvm_line(line::AbstractString)
    parts = split(strip(line))
    label = parse(Float64, parts[1])
    features = Dict{Int, Float64}()
    observed_max = 0
    for token in Iterators.drop(parts, 1)
        idx_text, value_text = split(token, ':', limit = 2)
        idx = parse(Int, idx_text)
        value = parse(Float64, value_text)
        features[idx] = value
        observed_max = max(observed_max, idx)
    end
    return label, features, observed_max
end

function verify_header(csv_path::AbstractString, feature_count::Int)
    open(csv_path, "r") do io
        header = readline(io)
        fields = split(chomp(header), ',')
        expected = ["label"; ["x$(i)" for i in 1:feature_count]]
        fields == expected || error("Header mismatch for $(basename(csv_path))")
    end
end

function verify_file(name::AbstractString)
    libsvm_path = joinpath(LIBSVM_DIR, name)
    csv_path = joinpath(LIBSVM_DIR, "$(name).csv")
    isfile(csv_path) || error("Missing CSV for $name")

    observed_max = 0
    open(libsvm_path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            _, _, line_max = parse_libsvm_line(line)
            observed_max = max(observed_max, line_max)
        end
    end

    feature_count = inferred_feature_count(name, observed_max)
    verify_header(csv_path, feature_count)

    checked_rows = 0
    open(libsvm_path, "r") do libio
        open(csv_path, "r") do csvio
            readline(csvio)
            for raw_line in eachline(libio)
                line = strip(raw_line)
                isempty(line) && continue

                eof(csvio) && error("CSV ended early for $name at row $(checked_rows + 1)")
                csv_line = chomp(readline(csvio))

                label, features, _ = parse_libsvm_line(line)
                fields = split(csv_line, ',')
                length(fields) == feature_count + 1 || error("Column count mismatch in $name row $(checked_rows + 1)")

                isempty(fields[1]) && error("Blank label in $name row $(checked_rows + 1)")
                parse(Float64, fields[1]) == label || error("Label mismatch in $name row $(checked_rows + 1)")

                for feature_idx in 1:feature_count
                    field = fields[feature_idx + 1]
                    isempty(field) && error("Blank x$(feature_idx) in $name row $(checked_rows + 1)")
                    csv_value = parse(Float64, field)
                    expected_value = get(features, feature_idx, 0.0)
                    csv_value == expected_value || error("Value mismatch in $name row $(checked_rows + 1), x$(feature_idx)")
                end

                checked_rows += 1
            end

            eof(csvio) || error("CSV has extra rows for $name")
        end
    end

    println("Verified $name: $checked_rows rows, $feature_count features")
end

function main()
    names = sort(filter(name -> !endswith(name, ".csv"), readdir(LIBSVM_DIR)))
    for name in names
        verify_file(name)
    end
end

main()
