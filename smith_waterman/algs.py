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

