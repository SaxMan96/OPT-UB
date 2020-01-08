# Exercise 5 - Mateusz Dorobek

Iterative conjugate Gradient Method with Ax = b, where matrix A is real, symetric and positive defined. 


$$
r_{0}:=b-Ax_{0}\\
p_{0}:=r_{0}\\
k:=0\\
$$
**repeat**

$$
\alpha _{k}:={\frac {r_{k}^{\top }r_{k}}{p_{k}^{\top }Ap_{k}}}\\
x_{k+1}:=x_{k}+\alpha _{k}p_{k}\\
r_{k+1}:=r_{k}-\alpha _{k}Ap_{k}\\
$$

**if** r is small enough
    **then** exit loop
**end if**
$$
\beta _{k}:={\frac {r_{k+1}^{\top }r_{k+1}}{r_{k}^{\top }r_{k}}}\\
p_{k+1}:=r_{k+1}+\beta _{k}p_{k}\\
k:=k+1
$$

**end repeat**

**Result is:** 
$$
x_{k+1}
$$


```python
from IPython.display import display, Math

def bvalue(var, a):
    display(Math(var+' = '+str(round(a,2))))
    
def bmatrix(var, a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    display(Math(var+' = '+'\n'.join(rv)))
```


```python
from IPython.display import display, Math

def bvalue(var, a):
    return str(var+' = '+str(round(a,2)))
    
def bmatrix(var, a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return str(var+' = '+'\n'.join(rv))
```


```python
import numpy as np

def conjgrad(A,b,x0):
    r = b - A@x0
    p = r
    x = x0
    disp = ""
    for k, v in {"A": A, "b": b, "x_{0}": x0, "r_{0}": r, "p_{0}": p}.items():
        disp += bmatrix(k,v) + ", "
    display(Math(disp))
    for i in range(A.shape[0]):
        print("Iteration: ", i)
        alpha = (r.T@r)/(p.T@(A@p))
        x = x + alpha*p
        r_new = r - alpha*(A@p)
        if np.linalg.norm(r) < 1e-10:
            break
        beta = (r_new.T@r_new)/(r.T@r)
        p = r_new + beta*p
        r = r_new
        disp = ''
        for k, v in {r"\alpha": alpha.item(), r"\beta": beta.item()}.items():
            disp += bvalue(k,v) + ", "
        for k, v in {"x_{"+str(i+1)+"}": x, "r_{"+str(i+1)+"}": r, "p_{"+str(i+1)+"}": p}.items():
            disp += bmatrix(k,v) + ", "
        display(Math(disp))
    return x
```


```python
A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
b = np.array([[1,1,1]]).T
x0 = np.array([[0,0,0]]).T
x = conjgrad(A,b,x0)
print("Result:")
display(Math(bmatrix("x",x)))
```


$$
\displaystyle A = \begin{bmatrix}
  1 & 0 & 0\\
  0 & 2 & 0\\
  0 & 0 & 3\\
\end{bmatrix}, b = \begin{bmatrix}
  1\\
  1\\
  1\\
\end{bmatrix}, x_{0} = \begin{bmatrix}
  0\\
  0\\
  0\\
\end{bmatrix}, r_{0} = \begin{bmatrix}
  1\\
  1\\
  1\\
\end{bmatrix}, p_{0} = \begin{bmatrix}
  1\\
  1\\
  1\\
\end{bmatrix}, 
$$


Iteration:  0



$$
\displaystyle \alpha = 0.5, \beta = 0.17, x_{1} = \begin{bmatrix}
  0.5\\
  0.5\\
  0.5\\
\end{bmatrix}, r_{1} = \begin{bmatrix}
  0.5\\
  0.\\
  -0.5\\
\end{bmatrix}, p_{1} = \begin{bmatrix}
  0.66666667\\
  0.16666667\\
  -0.33333333\\
\end{bmatrix}, 
$$


Iteration:  1



$$
\displaystyle \alpha = 0.6, \beta = 0.12, x_{2} = \begin{bmatrix}
  0.9\\
  0.6\\
  0.3\\
\end{bmatrix}, r_{2} = \begin{bmatrix}
  0.1\\
  -0.2\\
  0.1\\
\end{bmatrix}, p_{2} = \begin{bmatrix}
  0.18\\
  -0.18\\
  0.06\\
\end{bmatrix}, 
$$


Iteration:  2


$$
\displaystyle \alpha = 0.56, \beta = 0.0, x_{3} = \begin{bmatrix}
  1.\\
  0.5\\
  0.33333333\\
\end{bmatrix}, r_{3} = \begin{bmatrix}
  1.38777878e-17\\
  2.77555756e-17\\
  1.38777878e-17\\
\end{bmatrix}, p_{3} = \begin{bmatrix}
  1.38777878e-17\\
  2.77555756e-17\\
  1.38777878e-17\\
\end{bmatrix},
$$

Result:

$$
\displaystyle x = \begin{bmatrix}
  1.\\
  0.5\\
  0.33333333\\
\end{bmatrix}
$$


Mateusz Dorobek
