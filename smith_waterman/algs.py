import numpy as np

#functions associated with SW
def load_sim_matrix(name):
    pathway = os.path.join('..', name) #on Github, needs to be changed to '..'
    if name == 'BLOSUM50' or name == 'BLOSUM62':
        sim_matrix = np.loadtxt(os.path.abspath(pathway), dtype=int, comments='#', skiprows=7)
    elif name == 'MATIO':
        sim_matrix = np.loadtxt(os.path.abspath(pathway), dtype=int, comments='#', skiprows=3)
    else:
        sim_matrix = np.loadtxt(os.path.abspath(pathway), dtype=int, comments='#', skiprows=10)
        
    AA_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", 
           "M", "F", "P", "S", "T", "W", "Y", "V", "B", "Z", "X", "x"]
    similarity_matrix_index = {}
    for i, aa in enumerate(AA_list):
        similarity_matrix_index[aa] = i
    
    return sim_matrix, similarity_matrix_index

def matrix(a, b, match_score_similarity_matrix, similarity_matrix_index, GOC, EP):
    H = np.zeros((len(a) + 1, len(b) + 1), np.int) #initializing matrix
    flag = 0 #initialize flag to zero (no gap has been created yet)

    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])): #going through every cell in the matrix
        match = H[i - 1, j - 1] + match_score_similarity_matrix[similarity_matrix_index[a[i-1]], similarity_matrix_index[b[j-1]]] #if AAs align, return score based on similarity matrix
        if flag == 0: #assuming that there's no gap being built so would use gap opening cost
            delete = H[i - 1, j] - GOC #If choose to instead "delete" or skip over, then will go one to left and also have a penalty cost
            insert = H[i, j - 1] - GOC #If choose to instead "insert" (skipping over on other strand), then will be moving on the vertical axis with a penalty
        else: #assumes that there is a gap so need to use extension penalty instead
            delete = H[i - 1, j] - EP #If choose to instead "delete" or skip over, then will go one to left and also have a penalty cost
            insert = H[i, j - 1] - EP
        H[i, j] = max(match, delete, insert) #Whichever of the above 3 options is best/optimal, will be picked.
        if H[i,j] == match:
            flag = 0 #if aligned, then there is still no gap created
        else: flag = 1 #gap has been started and continuation penalty should be used instead of start penalty
        if H[i,j] < 0: H[i,j] = 0 #if negative, will instead return 0 because that's how S-WM works
    return H

def traceback(H, b, b_='', old_i=0):
    H_flip = np.flip(np.flip(H, 0), 1) #flip H to get the *last* (as in bottom right) occurrence of H.max() instead of top left
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))  #(i, j) equal the *last* indexes of H.max()/highest score in matrix
    if H[i, j] == 0: #terminating condition--which happens once you hit zero
        return b_, j
    b_ = b[j - 1] + '-' + b_ if old_i - i > 1 else b[j - 1] + b_ #Otherwise, add gap to sequence if True; otherwise, add aligned AA
    return traceback(H[0:i, 0:j], b, b_, i)

#score function
def score(H):
    H_flip = np.flip(np.flip(H, 0), 1)
    return H_flip.max()

def aligned_seq_score(alignment, match_score_similarity_matrix, similarity_matrix_index):
    score = 0
    flag = 0
    for character in alignment:
        if character == '-' and flag == 0: #it's the beginning of a gap
            added_value = -10 #give it gap opening penalty
            score = score + added_value
            flag = 1 #show we're now in the midst of a gap
        elif character == '-' and flag == 1: #we're in the middle of a gap
            added_value = -4 #give it extension penalty
            score = score + added_value
            flag = 1 #show we're still in the midst of a gap
        else: #in this case, we have a non-gap alignment, so give it the score specified in the similarity matrix
            added_value = match_score_similarity_matrix[similarity_matrix_index[character], similarity_matrix_index[character]]
            score = score + added_value
            flag = 0 #we're not in a gap
    return score


#functions associated with ROC
def return_threshold(similarity_matrix, GOC, EP, goal_TPR):
    pos_pairs_scores = []
    for f in pos_pairs_filelist:
        a, b = return_sequences(*f)
        score_matrix = matrix(a, b, similarity_matrix, BLOSUM50_i, GOC, EP) #calculates the score matrix for two sequences
        pos_pairs_scores.append(score(score_matrix)) #finds the score from the score matrix and appends to list
    pos_pairs_scores = np.array(pos_pairs_scores) #sorts list of scores in ascending order
    threshold = np.sort(pos_pairs_scores)[int(49 * (1 - goal_TPR))] #picks threshold as the score based on desired TPR
    return threshold

def find_FPR(threshold, similarity_matrix, GOC, EP):
    neg_pairs_scores = []
    for f in neg_pairs_filelist: #looping through all the negative sequence pairs
        a, b = return_sequences(*f)
        score_matrix = matrix(a, b, similarity_matrix, BLOSUM50_i, GOC, EP) #finding the score for each negative pair sequence
        neg_pairs_scores.append(score(score_matrix)) #adding each score to a matrix
    FPR = (np.array(neg_pairs_scores) > threshold).mean() #calculating the FPR by comparing all the scores to a threshold and averaging it
    return FPR

#optimization function
def return_threshold_FPR(similarity_matrix, goal_FPR, negative_alignment_list):
    neg_pairs_scores = []
    for alignment in negative_alignment_list:
        score = aligned_seq_score(alignment, similarity_matrix, similarity_matrix_index) #calculates the score matrix for two sequences
        neg_pairs_scores.append(score) #finds the score from the score matrix and appends to list
    neg_pairs_scores = np.array(neg_pairs_scores)
    threshold = np.sort(neg_pairs_scores)[int(49 * (1 - goal_FPR))] #picks threshold as the score based on desired TPR
    return threshold

def find_TPR(threshold, similarity_matrix, positive_alignment_list):
    pos_pairs_scores = []
    for alignment in positive_alignment_list:
        score = aligned_seq_score(alignment, similarity_matrix, similarity_matrix_index) #note all similarity matrix indices are the same
        pos_pairs_scores.append(score)
    TPR = (np.array(pos_pairs_scores) > threshold).mean()
    return TPR

def obj_value(sim_matrix, negative_alignment_list, positive_alignment_list):
    TPRs = []
    for goal_FPR in [0.0, 0.1, 0.2, 0.3]:
        threshold = return_threshold_FPR(sim_matrix, goal_FPR, negative_alignment_list)
        TPR = find_TPR(threshold, sim_matrix, positive_alignment_list)
        TPRs.append(TPR)
    TPRs = np.array(TPRs)
    sum_TPR = TPRs.sum()
    return sum_TPR

def grad_descent(sim_matrix, negative_alignment_list, positive_alignment_list, learning_rate=0.1):
    sim_matrix = sim_matrix.astype(float)
    
    
    AA_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", 
        "M", "F", "P", "S", "T", "W", "Y", "V", "B", "Z", "X", "x"]
    
    #updating the amino acids
    for iterations in range(50): #I would generally use more iterations, but the algorithm takes awhile to run and I got improvement with only 50, so I'm going with it.
        for AA in AA_list: #iterating through each--for ref, each "AA" refers to one of the values in matrix I'm trying to change
            #Calculate partial derivative of each "diagnol value"
            h=0.01 #small number to shift value of my matrix slightly
            new_matrix = sim_matrix #creating copy of sim_matrix that I can add "h" to in order to calculate the loss function when one value of the matrix is changed slightly
            new_matrix[similarity_matrix_index[AA], similarity_matrix_index[AA]] = new_matrix[similarity_matrix_index[AA], similarity_matrix_index[AA]] + h
            new_value = obj_value(new_matrix, negative_alignment_list, positive_alignment_list)
            new_cost = np.abs(4 - new_value) #value of loss function when one value of sim_matrix is shifted by small value "h"
        
            old_value = obj_value(sim_matrix, negative_alignment_list, positive_alignment_list)
            old_cost = np.abs(4 - old_value) #value of loss function with current sim_matrix
            p_derivative = (new_cost - old_cost)/h #calculation of partial derivative: dC/d(AA) = (L(AA + h) - L(AA))/h
            
            #my update: value = value - learning_rate * dC/d(AA)
            sim_matrix[similarity_matrix_index[AA], similarity_matrix_index[AA]] = sim_matrix[similarity_matrix_index[AA], similarity_matrix_index[AA]] - learning_rate * p_derivative
    
    
    return sim_matrix

