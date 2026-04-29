from modules.params import *


def compute_entropy(sequence):
    """
    Computes the Shannon entropy of a sequence.

    Parameters:
    sequence (str): The input sequence for which the entropy is to be computed.

    Returns:
    float: The Shannon entropy of the sequence.
    """
    # Calculate the frequency of each symbol in the sequence
    m = len(sequence)
    bases = collections.Counter([tmp_base for tmp_base in sequence])
 
    shannon_entropy_value = 0
    for base in bases:
        # number of residues
        n_i = bases[base]
        # n_i (# residues type i) / M (# residues in column)
        p_i = n_i / float(m)
        entropy_i = p_i * (math.log(p_i, 2))
        shannon_entropy_value += entropy_i
 
    return shannon_entropy_value * -1

def compute_bigram_entropy(sequence):
    """
    Computes the Shannon entropy of the probability distribution of bigrams (pairs of consecutive symbols)
    in a given sequence.

    Parameters:
    sequence (str): The input sequence.

    Returns:
    float: The Shannon entropy based on bigram probabilities.
    """
    if len(sequence) < 2:
        return 0.0  # Entropy is 0 for sequences too short to form bigrams

    # Step 1: Extract all overlapping bigrams
    bigrams = [sequence[i:i+2] for i in range(len(sequence) - 1)]

    # Step 2: Count occurrences of each unique bigram
    bigram_counts = collections.Counter(bigrams)

    # Step 3: Compute the probability distribution of bigrams
    total_bigrams = sum(bigram_counts.values())  # Total number of bigrams
    bigram_probs = {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}

    # Step 4: Compute Shannon entropy
    shannon_entropy_value = -sum(p * math.log2(p) for p in bigram_probs.values())

    return shannon_entropy_value

def lempel_ziv_complexity(sequence):
    """
    Computes the Lempel-Ziv complexity of a sequence.

    Parameters:
    sequence (str): The input sequence for which the Lempel-Ziv complexity is to be computed.

    Returns:
    int: The Lempel-Ziv complexity of the sequence.
    """
    n = len(sequence)
    i, k, l = 0, 1, 1
    complexity = 1

    while True:
        if sequence[i + k - 1] != sequence[l + k - 1]:
            if k > l:
                l = k
            i += 1
            if i == l:
                complexity += 1
                l += 1
                if l == n:
                    break
                i = 0
            k = 1
        else:
            k += 1
            if l + k > n:
                complexity += 1
                break

    return complexity


def AG_complexity(x):
    """
    Computes the "AG Complexity" of a binary sequence x.

    Parameters:
    x (list or np.array): A binary sequence (0s and 1s).

    Returns:
    float: The computed complexity value.
    
    The method constructs a matrix tracking differences between subsequences
    and calculates a weighted sum over its profile to quantify complexity.
    """

    # Length of the sequence
    L = len(x)

    # Edge cases: 
    # - If the sequence has only one element OR 
    # - If the sequence is entirely 0s or entirely 1s (homogeneous),
    #   then complexity is 0 (fully predictable structure).
    if L == 1 or np.sum(x) == 0 or np.sum(x) == L:
        return 0

    # Initialize an L × L matrix M with zeros
    M = np.zeros((L, L), dtype=int)

    # Step 1: Fill first column (j=1) of the matrix M
    # This column represents transitions between adjacent elements
    for i in range(L - 1):
        M[i, 1] = 1 if x[i] != x[i + 1] else 0  # Mark 1 if a transition occurs

    # Step 2: Fill the rest of matrix M (for increasing subsequence lengths j)
    for j in range(2, L):  # Iterate over subsequence lengths
        for i in range(L - j):  # Start positions in sequence
            for k in range(1, j):  # Compare elements at different steps in the subsequence
                if M[i, k] != M[i + j, k]:
                    M[i, j] = 1  # Mark as 1 if differences are found between subsequences

    # Step 3: Compute Profile
    # Profile sums M values for each subsequence length (excluding j=1)
    Profile = np.sum(M[:, 1:L], axis=0)

    # Step 4: Compute weighted sum of the Profile
    a = np.arange(2, L + 1)  # Indexes for sequence length starting from 2
    weights = 1 / (L - a + 1)  # Compute weighting factors for each subsequence length

    # Compute final complexity as weighted sum
    Complexity = np.sum(Profile * weights)

    return Complexity


def change_complexity_multisymbol(seq):
    changes = 0
    for i in range(len(seq) - 1):
        if seq[i] != seq[i+1]:
            changes += 1
    return changes


def algorithmic_complexity(sequence):
    """
    Computes an approximation of the algorithmic complexity of a sequence
    using compression-based methods.

    Parameters:
    sequence (str): The input sequence for which the algorithmic complexity is to be computed.

    Returns:
    int: The approximate algorithmic complexity of the sequence.
    """
    # Convert the sequence to bytes
    sequence_bytes = sequence.encode('utf-8')
    
    # Compress the sequence using zlib
    compressed_sequence = zlib.compress(sequence_bytes)
    
    # The length of the compressed sequence is used as an approximation of the algorithmic complexity
    complexity = len(compressed_sequence)
    
    return complexity


def chunk_complexity(sequence):
    """
    Computes the chunk complexity of a sequence based on the formula proposed by Mathy & Feldman.

    Parameters:
    sequence (str): The input sequence for which the chunk complexity is to be computed.

    Returns:
    float: The chunk complexity of the sequence.
    
    Chunkcomplexity ¼ PK i¼1 log2ð1 þ LiÞ,where Kisthe number of chunks and Li the length of the i-th run.
    """
    if not sequence:
        return 0

    chunks = []
    current_char = sequence[0]
    current_length = 1

    for char in sequence[1:]:
        if char == current_char:
            current_length += 1
        else:
            chunks.append(current_length)
            current_char = char
            current_length = 1
    chunks.append(current_length)

    complexity = sum(math.log2(1 + length) for length in chunks)
    
    return complexity

def is_symmetric(subsequence):
    """
    Check if a given subsequence is symmetric.
    
    Parameters:
    subsequence (str): The subsequence to check.
    
    Returns:
    bool: True if the subsequence is symmetric, False otherwise.
    """
    return subsequence == subsequence[::-1]

def count_subsymmetries(sequence):
    """
    Compute the number of symmetric sub-sequences within a given sequence.
    
    Parameters:
    sequence (str): The input sequence for which the number of subsymmetries is to be computed.
    
    Returns:
    int: The number of symmetric sub-sequences in the sequence.
    """
    n = len(sequence)
    subsymmetry_count = 0
    
    # Check all possible sub-sequences of the sequence
    for length in range(2, n + 1):
        for start in range(n - length + 1):
            subsequence = sequence[start:start + length]
            if is_symmetric(subsequence):
                subsymmetry_count += 1
    
    return subsymmetry_count


dict_shannon_entropy = {key: compute_entropy(value) for key, value in real_mapping.items()}
dict_shannon_entropy_bigram = {key: compute_bigram_entropy(value) for key, value in real_mapping.items()}
dict_lz_complexity={key: lempel_ziv_complexity(value) for key, value in real_mapping.items()}
dict_change_complexity={key: AG_complexity([int(i) for i in value]) for key, value in real_mapping.items()}
dict_change_complexity_multi={key: change_complexity_multisymbol([int(i) for i in value]) for key, value in real_mapping.items()}
dict_algorithmic_complexity={key: algorithmic_complexity(value) for key, value in real_mapping.items()}
dict_subsymetrie={key: count_subsymmetries(value) for key, value in real_mapping.items()}
dict_chunk_complexity_local={key:value for key,value in zip(real_mapping.keys(),chunk_comp_array)}
dict_chunk_complexity_global={key:value for key,value in zip(real_mapping.keys(),chunk_comp_array_global)}

