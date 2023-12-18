# Delvino Ardi - 1313621025
# Rawdo Madina - 1313621028

using Serialization, Statistics, LinearAlgebra

function avg_perclass(dataset)
    classes = unique(dataset[:, end])
    num_features = size(dataset, 2) - 1
    num_classes = length(classes)
    mu_vec = zeros(Float64, 1, num_features, num_classes)

    for i = 1:num_classes
        current_class = classes[i]
        current_class_indices = abs.(dataset[:, end] .- current_class) .< 0.1
        current_data = dataset[current_class_indices, 1:end-1]

        if size(current_data, 1) > 0
            current_data = Float64.(current_data)
            mu = mean(current_data, dims=1)
            mu_vec[1, :, i] = mu
        else
            mu_vec[1, :, i] .= 0.0
        end
    end

    return mu_vec
end

function d1_distance(features, mu_vector)
    num_classes = size(mu_vector, 3)
    features = repeat(features, outer=[1, 1, num_classes])
    subtracted_vector = abs.(features .- mu_vector)
    return subtracted_vector
end

function classify_by_distance(features, mu_vector)
    num_instances = size(features, 1)
    mu_vector = repeat(mu_vector, outer=[num_instances, 1, 1])
    dist_vector = d1_distance(features, mu_vector)
    min_vector = argmin(dist_vector, dims=3)
    min_index = @. get_min_index(min_vector)
    return min_index
end

function get_min_index(X)
    return X[3]
end

function cascade_classify(dataset, mu_vector)
    num_features = size(dataset, 2) - 1
    features = dataset[:, 1:num_features]
    num_instances = size(features, 1)

    preds = zeros(Int, num_instances, num_features)

    for i = 1:num_features
        current_feature = features[:, i]
        current_feature = reshape(current_feature, (num_instances, 1, 1))
        current_mu = reshape(mu_vector[1, i, :], (1, 1, size(mu_vector, 3)))
        current_pred = classify_by_distance(current_feature, current_mu)

        preds[:, i] = current_pred[:, 1, 1]
    end

    truth = dataset[:, end]
    temp = hcat(dataset, preds)
    return truth, preds, temp
end

function confusion_matrix(truth, preds)
    class = unique(truth)
    class_size = length(class)
    valuation = zeros(Int, class_size, class_size)
    
    for i = 1:class_size
        for j = 1:class_size
            valuation[i, j] = sum((truth .== class[i]) .& (preds .== class[j]))
        end
    end
    
    return valuation
end

function true_correctness(valuation)
    return sum(diag(valuation)) / sum(valuation)
end

dataset = deserialize("data_9m.mat")
mu_vector = avg_perclass(dataset)
truths, preds, temp = cascade_classify(dataset, mu_vector)
valuation = confusion_matrix(truths, preds)
accuracy = true_correctness(valuation)
display(valuation)
println("\n", accuracy)
