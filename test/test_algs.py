import itertools
import numpy as np
import os
import smith_waterman

#for some reason, my test file is not loading in any of my functions, 
#so I had to copy and paste all my functions from my algs file into here. I will document where the actual tests begin...

###COPY AND PASTE OF MY ALGS.PY FILE##########

#functions associated with SW
def load_sim_matrix(name):
    pathway = os.path.join('.', name) #on Github, needs to be changed to '..'?
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
pos_pairs_filelist = []
with open('./Pospairs.txt', mode='r') as my_file:
    for line in my_file:
        pos_pairs_filelist.append(line.strip().split(' '))
        
def return_sequences(sequence_a_path, sequence_b_path):
    with open (sequence_a_path, "r") as file: #creating string for sequence a from its filepath
        a = file.read()
        a = a[a.find("\n")+1:].replace('\n', '')
    with open (sequence_b_path, "r") as file: #creating string for sequence b from its filepath
        b = file.read()
        b = b[b.find("\n")+1:].replace('\n', '')
    return a, b #the two sequences I am going to compare/align using Smith-Waterman

def return_threshold(similarity_matrix, GOC, EP, goal_TPR):
    pos_pairs_scores = []
    for f in pos_pairs_filelist:
        a, b = return_sequences(*f)
        score_matrix = matrix(a, b, similarity_matrix, BLOSUM50_i, GOC, EP) #calculates the score matrix for two sequences
        pos_pairs_scores.append(score(score_matrix)) #finds the score from the score matrix and appends to list
    pos_pairs_scores = np.array(pos_pairs_scores) #sorts list of scores in ascending order
    threshold = np.sort(pos_pairs_scores)[int(49 * (1 - goal_TPR))] #picks threshold as the score based on desired TPR
    return threshold

neg_pairs_filelist = []
with open('./Negpairs.txt', mode='r') as my_file:
    for line in my_file:
        neg_pairs_filelist.append(line.strip().split(' '))

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



##################HERE IS WHERE THE ACTUAL TESTS BEGIN###############

#loading in my similarity matrix
BLOSUM50, BLOSUM50_i = load_sim_matrix("BLOSUM50")
BLOSUM50, similarity_matrix_index = load_sim_matrix("BLOSUM50")

def test_roc():
    ROC_TPR = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    #test that my FPR outputs are between 0 and 1
    for goal_TPR in ROC_TPR:
        threshold = return_threshold(BLOSUM50, 2, 5, goal_TPR)
        FPR = find_FPR(threshold, BLOSUM50, 10, 4)
        assert FPR >= 0.0
        assert FPR <= 1.0

def test_smithwaterman():
    
    #test that I get perfect alignment (alignment is the same as my sequence if sequences are the same)
    perfect_seq = "ARN"
    perfect_m = matrix(perfect_seq, perfect_seq, BLOSUM50, BLOSUM50_i, 2, 3)
    string_perfect, n_ = traceback(perfect_m, perfect_seq)
    assert string_perfect == perfect_seq

    #test that I get no alignment when my sequences are completely different
    bad_seq = "GZZ"
    bad_m = matrix(perfect_seq, bad_seq, BLOSUM50, BLOSUM50_i, 2, 3)
    string_bad, n_ = traceback(bad_m, bad_seq)
    assert string_bad == ""

def test_scoring():
    
    #test that my score function gives me the same value as my S-WM (at least for simple ones)
    perfect_seq = "ARN"
    perfect_m = matrix(perfect_seq, perfect_seq, BLOSUM50, BLOSUM50_i, 2, 3)
    assert aligned_seq_score("ARN", BLOSUM50, BLOSUM50_i) == score(perfect_m)

def test_optimization():
    
    #inserting the alignments I am using as my positive and negative ground truth
    pos_pairs_alignments = ['SLE-AK-R-Y-Y-N-P-A-QGQKI-LA-V', 'A-ED-AV-R-SAL-SP-P-QIK-G-EG-A-I-FKQK-Q', 'K-L-LEHILN-INN-K-T-K-T-E-C-LI', 'GAQ-N-C-ACH-G-NS-A-D-GGF-G-AMP-EIQAV', 'LND-Q-F-F-F-D-DGSI-DF-M-D-D-SWEEYL-M', 'G-KRK-R-GLGAS-ISKT', 'P-RT-ED-LHDAMV-D-G-PTLIQ-P-N-YQ-EP-P-P-N-Q-M-IG-W', 'EKV-AAK-F-NHSKF-D', 'F-PE-CSTT-F-D-G', 'P-PT-IAW-D-NFKV-GR-Q-Q-L', 'PD-P-I-WE-QYK-W-TS-RV', 'P-T-LCA-F-Q-G-VHV-D-E-E', 'I-GEA-T-G-ITLQ-CVH-AS-V-GSA-V-S-GKS-C-G-S', 'V-D-FLG-W-K-MT-K-H-S-D-T-G-K-V-D-L-T-E', 'DEGGM-F-PA-PGDSI-F-NV-GAD-K-TVGQE-VK-E-F-CAPH-GMVA-VVVG', 'MFQTF-N-K-T-P-I-V-L', 'KAI-K-AW-GC-A-I-L-N-SSER-P-G-N-KD-S', 'F-R-P-A-P-TD-RF-DT-T-L-T-Y-S-TAYNAL-P-A', 'EN-SQDT-TVHDG-P-S-LA-G-R-G-F-A-P-V-V-PTSS', 'RYA-IG-C-Y-V-P-SW-L-T-GN-NR-D-HY-GK-C-R-M-VG-FQ-P-I-DQ-IK-E', 'D-GS-PY-NS-S-F-WG-V-N-QNL-G-KQH-WE-E-D-G', 'ME-N-L-DRR-F-PFVT-P-ID-Q-AV-GA-IV', 'A-VKV-Y-L-GK-T-ERIL-YT-H-AA-M-F-D-Q-V-IRK-VF-K-VQL-D-V-D', 'A-KG-F-T-P-VN-V-TI-Y-DID-IL-ARQ-F-NK-DM-M', 'PY-I-YF-VRG-RMLL-DQGWKEE-DGD-QSNAILRHLGRS', 'LD-G-G-IRQV-S-KV-R-I-YGV-E-I-EQV-N-A', 'N-V-YW-GI-SS-KT-F-D-G-GS-PS-H-Q-Q-V', 'D-V-RE-WK-DNVV-Q-G-V-IKRL-T-NYD-DAFP', 'MK-A-PVRV-A-GAAG-G-Q-I-EGV-VM-LE-P-PD-D', 'YEI-G-IC-Q-A-A-WQ-YRE-LVE-LKE', 'G-S-G-S-A-Q-CALI-GGT-CVN-FDT-N', 'A-G-VVG-GTGG-T-AK-I-PS-V-L-I-G-D-G-D-GG-EF-R-G-K-I-A', 'D-GGA-L-A-ARA-GARVV-G-WQR-TGM-L', 'RVL-RM-G-GH-G-V-RI-E-GF-KVI-P-L', 'EA-V-Y-GG-G-G-S-VQAG-WA-S-W-D-I-N-R-VVT-KT-LT', 'LGG-I-GG-GL-A-VV-A-IA-L-E-YEN', 'RA-VVGGSG-G-TI-GF-GL-GF', 'DIRV-IA-GG-RQ-L-L-L-G-Q-D-LA-FD-H-D-I-F-N', 'AL-GL-GQNLI-A-N-G', 'SI-L-KN-T-KVI-G-F-G-F-AIA-G-PV', 'LG-AG-KIL-R-R-L-E-VTG-G-V-G', 'KGDEV-V-FVD-P-K-G-I-G-D-A-DVVV-R-D-I-ED-S', 'MP-AV-GAAG-AEM-D-E-G-VM-VA-V-V-L-A-L', 'RKVV-VGGG-GG-AA-LA-PSI-V-D-AS-PM-GKVAA-VV', 'KVSVVGAAG-VG-A-A-H-VVITAGI-RQPGQ-R-DL-NA-IMEDI-QS-PVD', 'RC-Y-N-GAS-G-CGVP-S-IA-A-IK-G', 'DQ-EVR-A-RGFM-V-P-H-A-D-YL-E-E-LFR-L', 'T-IL-DTARFRFLLGE-FSVP-NVHA-IIGEHGDTELPVWSQA-I-GV-P-EKKGAT-YG-A-GLARVTRAILHN-IL-SA-LDG-YG-EDV-GVPA-NGIREVIE-LND-EKN-A-LKS', 'G-C-N-GHC-C-C-GF-G-C', 'L-S-CSKCRK-C-DT-C-CRKNQY-N-C']
    neg_pairs_alignments = ['S-A-FD-EK-P-F-YN-RE-LQI-K', 'EP-CAG-H-G', 'NLEEV-KL-C-LVR-I', 'P-DSQ-IK-L-NVM-L-D-L', 'IE-K-C-C-GA-N-R-A-I-G-KC', 'P-G-V-SD-D-VV-G-RQ-KPGQS-L-A', 'ANHIISI-LNLV-F-E-LS-F-FSDS-L-K-SG-D-F-ET-Q', 'VEE-N-L-NQT-K', 'QR-GKK-WSNAG-P', 'DA-Q-IQ-V-SA-T', 'A-AP-D-A-GG-FFH-GS-HT-SS-G-S', 'PT-L-G-D-R', 'S-T-Q-GK-F-AL-I', 'M-ITQQA-PQV-AS-G-L-T-T-VA-PGD-DG-I-A-E-Y-K-GL', 'TPIE-L-D-E-WKITMNDGST-DL', 'F-LGDG-SFST-RG-W-E-G-SSL', 'MD-S-S-S-YS-NG-G-PY-V-D', 'MK-PVT-L-DVAE-S-QT-VNQ', 'P-FG-DA-IVL', 'KY-PQT-SG-FQ-G-GI-KIK-IG-P-E-Y-A', 'MRLG-D-LE-D-A-DSVAA-G-AAR-G-AV', 'METR-V-Y-WS-FFK-Q-EVG-MK', 'ID-GH-VDSL-V-C-Q-C-G-K-L', 'FVPGQ-ET-AE-G-D-G-E-A-VEM-G-F-P', 'QQ-I-G-G-G-VGC-PSA-Q-K-H-G-V-G-P', 'A-KP-I-Y-WI-KF', 'T-N-AA-VD-Y-V-EEV-AA-DA-A-L', 'Y-N-A-D-V-DL-RG-Q-K-FI-V-G-L-L-A-VA-LS', 'M-RL-PLT-AE-RIN-PFV-G-GQV-KHV', 'ER-VL-G-Q-RS-D-KRLA-HAQT-D-WA', 'M-QNI-GN-A-D-RA-L-G-F-DAI-SAV-DK-E-M-LAANGD', 'MKT-RI-N-F-A-D-ELV-V', 'AL-RVR-G-DL-G-P-G', 'ANNLF-SNGVSQ-AIL-VF-PSA-KHG', 'ML-LNS-E-RG-W', 'KR-V-ELER-Q-PEREH-P-IW-Y-MK-Q-K', 'SEK-DQ-C-CTK-P', 'R-R-V-D-G-L-EK-GVR-Y-VSE-Q-L-KIT', 'N-ML-D-T-SG-A-E-IA-K-CAV-S-P-V-I', 'V-RILN-KQ-P-AM-I-RQE-R-G', 'ZH-AD-P-NK', 'KI-S-F-G-G-G', 'LHYK-Q-LVD-P-IE-F', 'N-S-P-VT-HK-S-L-GS-K-NQ-F', 'E-V-YG-G-G-L-R-G-W-N-GKT', 'EKQ-F-E-IKD-VTK', 'PK-QSES-G-L-D-P-DAS-GD', 'K-RGS-CE-D-G', 'EEG-V-KFK-PK-CL-NT-C-P', 'LRSA-RWD-MA-D-S-G']
    
    new_BLOSUM50 = grad_descent(BLOSUM50, neg_pairs_alignments, pos_pairs_alignments)
    
    #test that the objective function value of BLOSUM50 is better after optimization than before
    assert obj_value(new_BLOSUM50, neg_pairs_alignments, pos_pairs_alignments) > obj_value(BLOSUM50, neg_pairs_alignments, pos_pairs_alignments)
    #test that my output is a matrix of the appropriate length
    assert new_BLOSUM50.shape == BLOSUM50.shape
    
