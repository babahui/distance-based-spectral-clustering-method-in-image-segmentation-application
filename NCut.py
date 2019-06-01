
def ncut(img=None, thresh=0.001, num_cuts=10, sp_met='slic'):
    from skimage import data, segmentation, color
    from skimage.future import graph
    
#     labels1 = segmentation.felzenszwalb(m_img, scale=50, sigma=0.5, min_size=100)
    if sp_met == 'slic':
        labels1 = segmentation.slic(img, compactness=30, n_segments=400)
    if sp_met == 'fl':
        labels1 = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    if sp_met == 'qs':
        labels1 = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    
    out1 = color.label2rgb(labels1, img, kind='avg')

    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g, thresh=thresh, num_cuts=num_cuts)
    out2 = color.label2rgb(labels2, img, kind='avg')
    
    return labels2