import numpy as np
import torch

def compute_coherence_matrix(scatter_matrix):
    """
    batch compute coherence matrix pixel wise from a scatter matrix 
    (no statistics for sliding window, just per pixel)

    scatter matrix: np array of complex numbers with shape [h,w,2,2]

    returns: coherence matrix of complex numbers with shape [h,w,3,3]       
    """
    
    sm = scatter_matrix
    # obtain kl = [Shh, sqrt(2) Shv, Svv]
    kl = np.r_[[sm[:,:,0,0], np.sqrt(2)*sm[:,:,0,1], sm[:,:,1,1]]]
    kl = np.transpose(kl, [1,2,0])

    # obtain coherence matrix for each pixel by doing kl * kl.T
    _kl = kl.reshape(*kl.shape, 1)                                # shape is [h,w,3,1]
    _klc = kl.conjugate().reshape(*kl.shape[:2], 1, kl.shape[2])  # shape is [h,w,1,3]
    cm = _kl*_klc                                                 # shape is [h,w,3,3]
    
    return cm

def compute_coherency_matrix_pauli(pauli_scattering_vector):
    """
    batch compute coherency matrix pixel wise from a Pauli scattering vector 
    (no statistics for sliding window, just per pixel)

    scatter pauli_scattering_vector: np array of complex numbers with shape [h,w,3]

    returns: coherency matrix of complex numbers with shape [h,w,3,3]       
    """
    
    wp = pauli_scattering_vector
    T = np.einsum("...i,...j->...ij", wp, wp.conj())
    
    return T

def generate_Pauli_RGB_from_T(T, scale_factor=2):
    """
    Generate a Pauli RGB image from a coherency matrix T

    Parameters
    ----------
    T : array of shape (h, w, 3, 3)
        Coherency matrix (covariance matrix of the Pauli scattering vector)
    scale_factor : float, optional
        Scaling factor of the RGB representation. The default is 2.

    Returns
    -------
    rgb : array of shape (h, w, 3)
        RGB representation in the range [0,1].

    """
    rgb = np.zeros(T.shape[:-2] + (3,))
    rgb[..., 0] = np.sqrt(T[...,1,1].real)
    rgb[..., 1] = np.sqrt(T[...,2,2].real)
    rgb[..., 2] = np.sqrt(T[...,0,0].real)
    pmean = np.mean(rgb, axis=(0,1))
    rgb /= pmean * scale_factor
    return rgb.clip(0, 1)

def get_H_A_alpha(T3, eps=1e-6):
    """
    Returns the H/A/alpha decomposition of a 3x3 coherency matrix T3.
    
    The decomposition is described in the paper:
    Cloude, S. R., & Pottier, E. (1996). A review of target decomposition
    theorems in radar polarimetry. IEEE transactions on geoscience and remote
    sensing, 34(2), 498-518.

    Parameters
    ----------
    T3 : ndarray of shape (..., 3, 3)
        Coherency matrix (covariance matrix of the Pauli scattering vector)
    eps : float, optional
        Small value to avoid numerical errors when computing the entropy for
        zero or very small eigenvalues. The default is 1e-6.

    Returns
    -------
    H : ndarray of shape (...)
        Entropy value for each pixel [0..1]
    A : ndarray of shape (...)
        Anisotropy value for each pixel [0..1]
    alpha : ndarray of shape (...)
        Mean alpha angle value for each pixel [0..np.pi/2]

    """
    l, V = np.linalg.eigh(T3)
    # Sort eigenvalues and eigenvectors in descending order, instead of ascending
    l = l[..., ::-1]
    V = V[..., ::-1]
    pi = l / np.sum(l,axis=-1)[..., np.newaxis] + eps
    H = np.sum(-pi*np.log(pi)/np.log(T3.shape[-1]), axis=-1)
    #H = scipy.stats.entropy(pi, base=T3.shape[-1])
    alphai = np.arccos(np.abs(V[...,0,:]))
    alpha = np.sum(pi*alphai, axis=-1)
    A = (pi[...,1] - pi[...,2]) / (pi[...,1] + pi[...,2])
    return H, A, alpha

def symmetric_revised_Wishart_distance(T1, T2, eps=1e-6):
    """
    Returns the Symmetric Revised Wishart distance (more precisely,
    dissimilarity measure) between two covariance matrices T1 and T2 pixelwise.
    
    It also applies a regularization factor based on eps.
    
    References:
    [1] A. Alonso-Gonzalez, C. Lopez-Martinez and P. Salembier, "Filtering and
    Segmentation of Polarimetric SAR Data Based on Binary Partition Trees,"
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 50, no. 2,
    pp. 593-605, Feb. 2012, doi: 10.1109/TGRS.2011.2160647
    [2] Qin, X., Zhang, Y., Li, Y., Cheng, Y., Yu, W., Wang, P., & Zou, H.
    (2022). Distance measures of polarimetric SAR image data: A survey.
    Remote Sensing, 14(22), 5873.

    Parameters
    ----------
    T1 : ndarray of shape (..., 3, 3)
        Covariance or coherency matrix of the first image.
    T2 : ndarray of shape (..., 3, 3)
        Covariance or coherency matrix of the second image.
    eps : float, optional
        Small value to avoid numerical errors when computing the dissimilarity
        for zero or very small eigenvalues. The default is 1e-6.

    Returns
    -------
    ndarray of shape (...)
        Symmetric Revised Wishart dissimilarity measure for each pixel.

    """
    T1 = T1 + np.eye(T1.shape[-1]) * eps
    T2 = T2 + np.eye(T2.shape[-1]) * eps
    return (np.sum(np.einsum("...ii->...i", np.linalg.solve(T1, T2)).real, axis=-1) + 
            np.sum(np.einsum("...ii->...i", np.linalg.solve(T2, T1)).real, axis=-1)
            ) / 2 - T1.shape[-1]



class AvgPool2dComplex(torch.nn.Module):
    
    """
    average pooling for complex numbers since `torch.nn.AvgPool2d` does not
    support complex numbers.
    
    since `torch.nn.Conv2d` **does** support them, this pooling is implemented
    with a convolution.
    """
    
    def __init__(self, n_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.conv = torch.nn.Conv2d(in_channels=n_channels, 
                                    out_channels=n_channels, 
                                    groups=n_channels, 
                                    kernel_size=kernel_size, 
                                    stride=kernel_size, 
                                    dtype=torch.cfloat,
                                    padding=0)
        
        # weight is one
        self.conv.weight.data = torch.ones_like(self.conv.weight.data).type(torch.cfloat)
        
        # bias is zero
        self.conv.bias.data   = torch.zeros_like(self.conv.bias.data).type(torch.cfloat)
        
        self.conv = self.conv.requires_grad_(False)
        
    def forward(self, x):
        x = self.conv(x)
        
        # mean pooling dividing by the window size
        x = x / self.kernel_size**2
        return x
    

def avgpool2D_complex(x, window_size):
    
    """
    convenience function to use AvgPool2dComplex with numpy arrays 
    (since AvgPool2dComplex expects 'channels first')

    Parameters
    ----------
    x: a numpy array of shape [h,w,n,n] (with n**2 channels last)
                     or shape [h,w,n] (with n channels last)

    window_size: the size of the sliding for averaging

    Returns
    -------
    a numpy array of shape [h//ws, w//ws, n, n] or [h//ws, w//ws, n], 
    with ws = window_size
    """
    
    # as tensor with shape [9, h, w]
    x_init = x
    n_channels = np.product(x.shape[2:])
    x = torch.tensor(np.transpose(x.reshape(*x.shape[:2],-1), [2,0,1])).type(torch.cfloat)

    # average with a window size 
    x = AvgPool2dComplex(kernel_size=window_size, n_channels=n_channels)(x).detach().numpy()
    x = np.transpose(x, [1,2,0]).reshape(*x.shape[-2:], *x_init.shape[2:])
    return x

def compute_quadpol_normalized_coherence_matrix(sm, window_size):
    """
    computes a normalized coherence matrix from a quadpol scatter matrix

    Parameters
    ----------
    sm: a np array scatter matrix of shape [h,w,2,2] with entries as complex numbers
        
    Returns
    -------
    ndarray of shape [h,w,9] where for each pixel you get 9 real numbers:
    
        d1, d2, d3, rho12.real, rho12.imag, rho13.real, rho13.imag, rho23.real, rho23.imag
    
    """
    shh = sm[:,:,:1,:1]
    shv = sm[:,:,:1,1:2]
    svv = sm[:,:,1:2,1:2]

    shh2 = avgpool2D_complex(shh*shh.conjugate(), window_size=window_size)
    shv2 = avgpool2D_complex(shv*shv.conjugate(), window_size=window_size)
    svv2 = avgpool2D_complex(svv*svv.conjugate(), window_size=window_size)
    shhhv = avgpool2D_complex(shh*shv.conjugate(), window_size=window_size)
    shhvv = avgpool2D_complex(shh*svv.conjugate(), window_size=window_size)
    shvvv = avgpool2D_complex(shv*svv.conjugate(), window_size=window_size)

    P = shh2 + 2*shv2 + svv2

    if not np.allclose(P.imag, 0, atol=1e-5):
        raise ValueError("P should have no imaginary part")

    d1 = (shh2/P).real
    d2 = 2*(shv2/P).real
    d3 = (svv2/P).real
    rho12 = shhhv / np.sqrt(shh2*shv2)    
    rho13 = shhvv / np.sqrt(shh2*svv2)   
    rho23 = shvvv / np.sqrt(shv2*svv2)    
        
    r = np.transpose(np.r_[[d1, d2, d3, rho12.real, rho12.imag, rho13.real, rho13.imag, rho23.real, rho23.imag]], [1,2,3,4,0]).reshape(*d1.shape[:2], -1)
    return r

def normalize_quadpol_coherence_matrix(x, window_size):
    """
    normalizes a quad pol coherence matrix
    
    Parameters
    ----------
    x: a coherence matrix of shape [h,w,3,3] with entries as complex numbers
    
    Returns
    -------
    ndarray of shape [h,w,9] where for each pixel you get 9 real numbers:
    
        d1, d2, d3, rho12.real, rho12.imag, rho13.real, rho13.imag, rho23.real, rho23.imag
    
    """
    
    # windoed average
    x = avgpool2D_complex(x, window_size = window_size)

    # P is the diagonal of the covariance at each averaged pixel
    P = np.einsum("abii->abi", x).sum(axis=-1)

    # d1 = <Shh^2> / P
    d1 = x[:,:,0,0] / P

    # d2 = 2*<Shv^2> / P
    d2 = x[:,:,1,1] / P

    # d3 = 2*<Shv^2> / P
    d3 = x[:,:,2,2] / P

    rho12 = x[:,:,0,1] / np.sqrt(2) / np.sqrt(x[:,:,0,0] * x[:,:,1,1]/2)

    rho13 = x[:,:,0,2] / np.sqrt(x[:,:,0,0] * x[:,:,2,2])

    rho23 = x[:,:,1,2] / np.sqrt(2) / np.sqrt(x[:,:,1,1] * x[:,:,2,2]/2)
    
    
    r = np.r_[[d1.real, d2.real, d3.real, rho12.real, rho12.imag, rho13.real, rho13.imag, rho23.real, rho23.imag]]
    r = np.transpose(r, [1,2,0])
    return r