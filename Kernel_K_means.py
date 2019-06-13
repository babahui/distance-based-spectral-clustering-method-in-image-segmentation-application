''' 
References:

Kernel k-means, Spectral Clustering and Normalized Cuts. Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis. KDD 2004.

Fast Global Alignment Kernels. Marco Cuturi. ICML 2011.
'''


from skimage.future import graph
import networkx as nx
from scipy import sparse

from tslearn.clustering import GlobalAlignmentKernelKMeans


def Kernel_K_Means(img_path, sp_met='slic', num_cuts=3):
    
    m_img = imread(img_path)
        
    # superpixels method
    if sp_met == 'felzenszwalb':
        segments = felzenszwalb(m_img, scale=10, sigma=0.5, min_size=100)
    elif sp_met == 'slic':
        segments = slic(m_img, compactness=30, n_segments=400)
    else:
        warnings.warn("Warning Message: no superpixels method parameter")
        
    # image -> eigenvectors
    g = graph.rag_mean_color(m_img, segments)
    w = nx.to_scipy_sparse_matrix(g, format='csc')
    entries = w.sum(axis=0)
    d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
    m = w.shape[0]
    d2 = d.copy()
    d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
    matrix = d2 * (d - w) * d2

    # matrix eigen-decomposition, scipy.sparse.linalg
    vals, vectors = scipy.sparse.linalg.eigsh(matrix, which='SM', k=min(100, m - 2))
    vals, vectors = np.real(vals), np.real(vectors)
    
    # get first K eigenvectors
    
#     index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
#     ev1, ev, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
    index = np.argsort(vals)[1]
    X_train = vectors[:, index]
    
    # Kernel K-means
    gak_km = GlobalAlignmentKernelKMeans(n_clusters=num_cuts, sigma=sigma_gak(X_train), n_init=20, \
                                         verbose=True, random_state=seed)
    sp_label = gak_km.fit_predict(X_train)
    
    # get pixel label
    p_label, labels = pixels_label(m_img, segments, sp_label)

    return p_label


def before_method(img_path, sp_met='felzenszwalb', graph_met='syn_met', admm_met='admm', num_cuts=3, dist_hist=False, lambda_coff=False, n_iter=1000):
    m_img = imread(img_path)

