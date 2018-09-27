import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from functools import reduce
from matplotlib import ticker, cm

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0 for i in range(3)]

def xy2bc(xy, tol=1.e-3):
  '''Converts 2D Cartesian coordinates to barycentric.
  Arguments:
    xy: A length-2 sequence containing the x and y value.
  '''
  s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75 for i in range(3)]
  return np.clip(s, tol, 1.0 - tol)

class Dirichlet(object):
  def __init__(self, alpha):
    '''Creates Dirichlet distribution with parameter `alpha`.'''
    from math import gamma
    from operator import mul
    self._alpha = np.array(alpha)
    self._coef = gamma(np.sum(self._alpha)) /reduce(mul, [gamma(a) for a in self._alpha])
  def pdf(self, x):
    '''Returns pdf value for `x`.'''
    from operator import mul
    return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])
  def sample(self, N):
    '''Generates a random sample of size `N`.'''
    return np.random.dirichlet(self._alpha, N)

def draw_pdf_contours(dist, nlevels=100, subdiv=8, **kwargs):
  '''Draws pdf contours over an equilateral triangle (2-simplex).
  Arguments:
    dist: A distribution instance with a `pdf` method.
    border (bool): If True, the simplex border is drawn.
    nlevels (int): Number of contours to draw.
    subdiv (int): Number of recursive mesh subdivisions to create.
    kwargs: Keyword args passed on to `plt.triplot`.
  '''
  refiner = tri.UniformTriRefiner(_triangle)
  trimesh = refiner.refine_triangulation(subdiv=subdiv)
  pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

  plt.tricontourf(trimesh, pvals, nlevels, cmap=cm.Blues, **kwargs)
  plt.axis('equal')
  plt.xlim(0, 1)
  plt.ylim(0, 0.75**0.5)
  plt.axis('off')

alphas = [[0.5] * 3, [1] * 3, [10] * 3, [2, 5, 10]]
for (i, alpha) in enumerate(alphas):
  plt.subplot(2, 2, i + 1)
  dist = Dirichlet(alpha)
  draw_pdf_contours(dist)
  plt.title(r'$\alpha$ = ({:.1f}, {:.1f}, {:.1f})'.format(*alpha), fontsize=16)
  
plt.savefig('img702.png', dpi=300, figsize=[5.5, 5.5])
