def _get_weight(choices, prob):
    # prob = min(prob, 1-1e-10)   # Prevent bugs, but this step is not required in probability
    split_value = 1 / (len(choices) - 1)
    idx = 0
    while(prob > split_value):
        idx += 1
        prob -= split_value
    relate_distance = prob / split_value
    return choices[idx] * (1 - relate_distance) + choices[idx + 1] * relate_distance

if __name__ == "__main__":
    rec_choice = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    ind_choice = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    cand = [0.7053411308078107, 0.9542592593410837, 0.9296478828883573, 0.28425047269448145, 0.1600125621449342, 0.47495464861462977]
    num_layers = int(len(cand) / 2)
    rec_weights = [0 for _ in range(num_layers)]
    ind_weights = [0 for _ in range(num_layers)]
    for i in range(0, 2 * num_layers, 2):
        rec = cand[i]
        ind = cand[i + 1]
        rec_weight = _get_weight(rec_choice, rec)
        ind_weight = _get_weight(ind_choice, ind)
        rec_weights[int(i/2)] = rec_weight
        ind_weights[int(i/2)] = ind_weight
    print(rec_weights, ind_weights)