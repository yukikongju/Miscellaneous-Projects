"""
user_mix: [num_mixes_listening x num_sounds]
[sounds] co_occurence: user_mix.T @ user_mix; co_occurence.fill_diagonal(0) [num_sounds x num_sounds]
   => co_occ(sound1, sound2) = lookup table
[mix] co_occurence(mix, sound): max/mean of each sounds co-occurence
    > query_mix = mix @ co_occ = [0, 0, 1, 0] @ co_occ; normalize

=> score = jaccard similarity (penalize frequent soinds naturally)
# |A ∩ B| / |A ∪ B|
intersection = co_occurrence[i, j]
union = co_occurrence[i, i] + co_occurrence[j, j] - intersection
score = intersection / union

=> PMI (pointwise mutual information) best for discovery

Notes:
- ALS learns latent structure whereas co-occurence matrix doesn't
    * A - B - C => co-occ[A, C] = 0 whereas als[A, C] > 0

"""
