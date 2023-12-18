# Delvino Ardi - 1313621025
# Rawdo Madina - 1313621028

using Serialization, Statistics, LinearAlgebra

dataset = deserialize("data_9m.mat")
features = dataset[:,1:end-1]
labels = dataset[:,end]

function fastest_avg_perclass(dataset)
    class = unique(dataset[:,end])
    class_index = size(dataset)[2]
    col_size = size(dataset)[2]
    feature_size = col_size-1
    class_size = length(class)
    mu_vec = zeros(Float16, 1, feature_size, class_size)
    for i = 1:class_size
        c = class[i]
        current_class_pos = (dataset[:, class_index] .- c) .< Float16(0.1)
        current_df = dataset[current_class_pos,1:class_index-1]
        current_df = Float32.(current_df)
        mu = mean(current_df, dims = 1)
        mu_vec[1,:,i] = mu
    end
    return mu_vec
end

mu_vector = fastest_avg_perclass(dataset)

function vectorized_d1_distance(X, mu)
    numclass = size(mu)[3]
    X = repeat(X, outer = [1,1,numclass])
    subtracted_vector = abs.(X .- mu)
    return subtracted_vector
end

function classify_by_distance_features(X, mu)
    num_instance = size(X)[1]
    mu_vec = repeat(mu, outer = [num_instance, 1, 1])
    dist_vec = vectorized_d1_distance(X, mu_vec)
    min_vector = argmin(dist_vec, dims=3)
    min_index = @.get_min_index(min_vector)
    return min_index
end

function get_min_index(X)
    return X[3]
end

function cascade_classify(dataset, mu)
    class = unique(dataset[:,end])
    class_index = size(dataset)[2]
    col_size = size(dataset)[2]
    feature_size = col_size-1
    preds = zeros(Int, size(dataset)[1], feature_size)
    for i = 1:feature_size
        current_feature = dataset[:,i]
        current_feature = reshape(current_feature, (size(current_feature)[1], 1))
        current_mu = reshape(mu[1,i,:], (1,1,size(mu)[3]))
        current_pred = classify_by_distance_features(current_feature, current_mu)
        if i == 1
            preds = current_pred
        else
            preds = hcat(preds, current_pred)
        end
    end
    truth = dataset[:, class_index]
    temp = hcat(dataset, preds)
    return truth, preds
end

function confusion_matrix(truth, preds)
    class = unique(truth)
    class_size = length(class)
    valuation = zeros(Int, class_size, class_size)
    for i = 1:class_size
        for j = 1:class_size
            valuation[i,j] = sum((truth .== class[i]) .& (preds .== class[j]))
        end
    end
    return valuation
end

function true_correctness(valuation)
    return sum(diag(valuation)) / sum(valuation)
end
truths, preds = cascade_classify(dataset, mu_vector)
valuation = confusion_matrix(truths, preds)
display(valuation)

correctness = true_correctness(valuation)
println(correctness)