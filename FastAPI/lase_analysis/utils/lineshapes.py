import numpy as np

def f_gaussian1(x, A, mu, fwhm, sat):
  return np.minimum(A*np.exp(-0.5*(2.355*(x-mu)/fwhm)**2), sat)

def f_gaussian2(x, A1, mu1, fwhm1, A2, mu2, fwhm2, sat):
  return np.minimum(A1*np.exp(-0.5*(2.355*(x-mu1)/fwhm1)**2) + A2*np.exp(-0.5*(2.355*(x-mu2)/fwhm2)**2), sat)
