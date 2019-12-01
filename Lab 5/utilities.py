import numpy as np
import scipy
from scipy import linalg
import time

#for all

def Newton_step(lamb0, dlamb, s0, ds):
    alp=1;
    idx_lamb0 = np.array(np.where(dlamb<0))
    if idx_lamb0.size > 0:
        alp = min(alp, np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size > 0:
        alp = min(alp, np.min(-s0[idx_s0]/ds[idx_s0]))
    return alp


#STRATEGIES

def s0():
    return matrix_KKT, rhv_small_KKT, dz_KKT
def s1():
    return matrix_s1, rhv_small_s1, dz_s1
def s2():
    return matrix_s2, rhv_small_s2, dz_s2

#step size function
def  step_size(z, matrix, rhv, solver, strategy, G, A, C, g, b, d, n, m, p):
    matrix_update, rhv_small_update, dz_update = strategy()
    epsilon = 1e-16
    lbd = z[-2*m:-m]
    s = z[-m:]
    rhv_small = rhv_small_update(z, rhv, G, A, C, g, d, n, p, m)
    #predictor substep
    dz_small = solver(matrix, rhv_small)
    dz = dz_update(z, dz_small, rhv, G, A, C, n, m, p)
    #step-size correction substep
    dlbd = dz[-2*m:-m]
    ds = dz[-m:]
    alpha = Newton_step(lbd, dlbd, s, ds)
    #3things
    mu = np.matmul(s, lbd)/m
    mu_tilda = np.matmul(s+alpha*ds, lbd+alpha*dlbd)/m
    sigma = (mu_tilda/mu)**3
    #corrector substep
    rhv[-m:] = rhv[-m:] - ds*dlbd + sigma*mu
    rhv_small = rhv_small_update(z, rhv, G, A, C, g, d, n, p, m)
    dz_small = solver(matrix, rhv_small)
    dz = dz_update(z, dz_small, rhv, G, A, C, n, m, p)
    #step-size correction substep
    dlbd = dz[-2*m:-m]
    ds = dz[-m:]
    alpha = Newton_step(lbd, dlbd, s, ds)
    #update substep
    z = z + 0.95*alpha*dz
    matrix = matrix_update(z, G, A, C, n, m, p)
    rhv = rhv_update(z, G, A, C, g, b, d, n, m, p)
    norm_rL = np.linalg.norm(rhv[:n])
    norm_rC = np.linalg.norm(rhv[n:n+p])
    if norm_rL<epsilon:
        #print('norm_rL')
        return False, z, matrix, rhv
    if norm_rC<epsilon and rhv[n:n+p].size>0 :
        #print('rC:{} shape:{} norm_rC:{}'.format(rhv[n:n+p],rhv[n:n+p].shape,norm_rC))
        return False, z, matrix, rhv
    if  abs(mu)<epsilon:
        #print('mu')
        return False, z, matrix, rhv
    return True, z, matrix, rhv

#optimization solver
def optimization(n, problem, strategy, solver, cond_nb=False):
    #initialize dimensions, matrices and vectors
    z, G, A, C, g, b, d, n, m, p = problem(n)
    #choose strategy functions
    matrix_update, _, _ = strategy()
    matrix = matrix_update(z, G, A, C, n, m, p)
    rhv = rhv_update(z, G, A, C, g, b, d, n, m, p)
    #loop variables
    niter = 25
    loop = True
    iloop=0
    time_list = []
    if cond_nb == True : cond_list = []
    while loop and iloop<niter:
        t0 = time.clock()
        loop, z, matrix, rhv = step_size(z, matrix, rhv, solver, strategy, G, A, C, g, b, d, n, m, p)
        t = time.clock() - t0
        time_list.append(t)
        iloop = iloop+1
        if cond_nb == True:
            cond_list.append(np.linalg.cond(matrix))

    #print('n:{} iters:{} z+g:{}'.format(n,iloop,np.linalg.norm(z[:n]+g)))
    #eq, ineq = constraint(z, A, C, b, d, n)
    #print('i={} f = {} eq:{} ineq:{} x_norm:{} loop:{}'.format(iloop,f(z, G, g, n), eq, ineq, np.linalg.norm(z[:n]),loop))
    t_total = sum(time_list)
    if cond_nb == True: return  t_total, iloop, np.array(cond_list)
    else: return t_total, iloop



def f(z, G, g, n):
    x = z[:n]
    return 0.5*np.matmul(x, G.dot(x)) + g.dot(x)

def constraint(z, A, C, b, d, n):
    x = z[:n]
    eq = np.all( np.matmul(A.T, x) - b < 1e-8 )
    ineq = np.all( np.matmul(C.T,x) - d  > 1e-8)
    return eq, ineq

def rhv_update(z, G, A, C, g, b, d, n, m, p):
    N = n + p + 2*m
    x = z[:n]
    gmm = z[n:n+p]
    lbd = z[-2*m:-m]
    s = z[-m:]
    F = np.zeros(N)
    F[:n] = np.matmul(G, x) + g - np.matmul(A, gmm) - np.matmul(C, lbd)
    F[n:n+p] = b - np.matmul(A.T, x)
    F[-2*m:-m] = s + d - np.matmul(C.T, x)
    F[-m:] =  s*lbd
    return -F

#STRATEGY 0 or Base

def matrix_KKT(z, G, A, C, n, m, p):
    N = n + p + 2*m
    KKT = np.zeros((N, N))
    KKT[0:n , 0:n] = G
    KKT[n:n+p , 0:n] = -A.T
    KKT[0:n , n:n+p] = -A
    KKT[n+p:n+p+m , 0:n] = -C.T
    KKT[0:n , n+p:n+p+m] = -C
    KKT[-2*m:-m , -m:] = np.eye(m)
    KKT[-m:, -2*m:-m] = np.diag(z[-m:])
    KKT[-m: , -m:] = np.diag(z[-2*m:-m])
    return KKT

def rhv_small_KKT(z, rhv, G, A, C, g, d, n, p, m):
    return rhv

def dz_KKT(z, dz_small, rhv, G, A, C, n, m, p):
    return dz_small


#STRATEGY 1

def matrix_s1(z, G, A, C, n, m, p):
    lbd = z[-2*m:-m]
    s = z[-m:]
    matrix = np.zeros((n+p+m, n+p+m))
    matrix[:n, :n] = G
    matrix[:n, n:-m] = -A
    matrix[n:-m, :n] = -A.T
    matrix[:n, -m:] = -C
    matrix[-m:, :n] = -C.T
    matrix[-m:, -m:] = np.diag(-s/lbd)
    return matrix


def rhv_small_s1(z, rhv, G, A, C, g, d, n, p, m):
    r3 = - rhv[-2*m:-m]
    r4 = - rhv[-m:]
    lbd = z[-2*m:-m]
    small_rhv = np.zeros(n+m+p)
    small_rhv[:n+p] = rhv[:n+p]
    small_rhv[-m:] = - r3 + r4/lbd
    return small_rhv

def dz_s1(z, dz_small, rhv, G, A, C, n, m, p):
    N = n + p + 2*m
    lbd = z[-2*m:-m]
    s = z[-m:]
    r4 = - rhv[-m:]
    dz = np.zeros(N)
    dz[:-m] = dz_small
    dz[-m:] = - ( r4 + s*dz[-2*m:-m])/lbd
    return dz

#STRATEGY 2

def matrix_s2(z, G, A, C, n, m, p):
    lbd = z[-2*m:-m]
    s = z[-m:]
    matrix = G + np.matmul(C/s*lbd,C.T)
    matrix = np.array(matrix)
    return matrix

def rhv_small_s2(z, rhv, G, A, C, g, d, n, p, m):
    lbd = z[-2*m:-m]
    s = z[-m:]
    r1 = -rhv[:n]
    r2 = -rhv[-2*m:-m]
    r3 = -rhv[-m:]
    small_rhv = -r1 - np.matmul(-C/s,-r3+ r2*lbd)
    return small_rhv

def dz_s2(z, dz_small, rhv, G, A, C, n, m, p):
    N = n + p + 2*m
    lbd = z[-2*m:-m]
    s = z[-m:]
    r2 = -rhv[-2*m:-m]
    r3 = -rhv[-m:]
    dz = np.zeros(N)
    dz[:n] = dz_small
    dz[-2*m:-m] =  (-r3 + lbd*r2)/s - np.matmul(C.T,dz_small)*lbd/s
    dz[-m:] = -r2 + np.matmul(C.T,dz_small)
    return dz

#LINEAR ALGEBRA SYSTEM SOLVING STRATEGIES

def gepp_solver(A,b):
    return np.linalg.solve(A,b)

def ldl_solver(A,b):
    n = len(A)
    L,D = ldl(A) #my function
    DLTX = scipy.linalg.solve_triangular(L, b, lower=True)
    LTX = np.array([DLTX[i]/D[i,i] for i in range(n)])
    dz  = scipy.linalg.solve_triangular(L.T, LTX, lower=False)
    return dz

def pseudo_ldl_solver(A,b):

    def block_diagonal_solver(D,b):
        n = len(D)
        x = np.zeros(n)
        i_to_pass = n
        for i in range(n):
            if i == i_to_pass:
                pass
            else:
                if i==n-1 or abs(D[i,i+1])<1e-16:
                    x[i] = b[i]/D[i,i]
                else:
                    x[i:i+2] = np.linalg.solve(D[i:i+2,i:i+2],b[i:i+2])
                    i_to_pass = i+1
        return x

    n = len(A)
    L,D,perm = scipy.linalg.ldl(A)
    L = L[perm]
    antiperm = np.zeros(n).astype('int')
    for i in range(n):
        antiperm[perm[i]] = i
    LDLTPTX = b[perm]
    DLTPTX = scipy.linalg.solve_triangular(L,LDLTPTX,lower=True)
    LTPTX = block_diagonal_solver(D,DLTPTX)
    PTX = scipy.linalg.solve_triangular(L.T,LTPTX,lower=False)
    x = PTX[antiperm]
    return x

def chol_solver(A,b):

    def chol(A):
        L, D = ldl(A)
        d = np.sqrt([D[i, i] for i in range(len(D))])
        G = np.array([L[i] * d[i] for i in range(len(d))])
        GT = L.T * d
        return G, GT

    G,GT = chol(A)
    GTX= scipy.linalg.solve_triangular(G, b, lower=True)
    dz = scipy.linalg.solve_triangular(GT, GTX, lower=False)
    return dz

def cho_scipy_solver(A, b):
    c, low = scipy.linalg.cho_factor(A)
    x = scipy.linalg.cho_solve((c, low), b)
    return x

def ldl(A):
    A = A.astype('float')
    n = len(A)
    for i in range(n-1):
        Aii  = A[i,i]
        if abs(Aii)<1e-10:
            print('danger')
            return
        for j in range(i+1,n):
            value =  A[j,i]/Aii
            A[j,i] = value
        for j in range(i+1,n):
            for k in range(j,n):
                A[k,j] = A[k,j] - A[j,i]*A[k,i]*Aii
    L = np.tril(A,-1) + np.eye(n)
    D = np.diag(np.diag(A))
    return L, D


def gen_lu_fixedblock_solver(nb):
    def lu_fixedblock(A, b):
        return lu_block_solver(A, b, nb)
    return lu_fixedblock



def lu_block_solver(A, b, nb):
    L,U = lu_block(A,nb)
    UX = scipy.linalg.solve_triangular(L, b, lower=True)
    x = scipy.linalg.solve_triangular(U, UX, lower=False)
    return x

def lu_block(A,nb):

    def lu_np(A):
    #Without pivoting
        n = len(A)
        A =A.astype('float')
        for i in range(n-1):
            aii = A[i,i]
            for j in range(i+1,n):
                A[j,i] = A[j,i]/aii
            for j in range(i+1,n):
                for k in range(i+1,n):
                    A[j,k] = A[j,k]-A[j,i]*A[i,k]
        L = np.tril(A,-1)+np.eye(n)
        U = np.triu(A)
        return L,U

    n = len(A)
    bl = n // nb
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for i in range(nb):
        id1 = bl*i
        id2 = bl*(i+1)
        A11 = A[:bl, :bl]
        A12 = A[:bl, bl:]
        A21 = A[bl:, :bl]
        A22 = A[bl:, bl:]
        #STEP 1 L11,U11
        L11,U11 = lu_np(A11)
        L[id1:id2, id1:id2]  = L11
        U[id1:id2, id1:id2]  = U11
        #STEP 2 U12
        U12 = np.linalg.solve(L11,A12)
        U[id1:id2, id2:] = U12
        #STEP 3 l21
        L21 = (np.linalg.solve(U11.T,A21.T)).T
        L[id2:, id1:id2] = L21
        #STEP 4 A22
        A = A22 - L21.dot(U12)
    return L,U

#READING VECTOR MATRICES

def read_matrix(source,shape,symm = False):
    matrix = np.zeros(shape)
    with open(source,'r') as file:
        a = file.readlines()
    for line in a:
        row, column, value = line.strip().split()
        row = int(row)
        column = int(column)
        value = float(value)
        matrix[row-1,column-1] = value
        if symm == True:
            matrix[column-1, row-1] = value
    return matrix

def read_vector(source,n):
    v = np.zeros(n)
    with open(source,'r') as file:
        a = file.readlines()
    for line in a:
        idx,value = line.strip().split()
        idx = int(idx)
        value = float(value)
        v[idx-1] = value
    return v

#initialization problem matrices and vectors and dimensions

def prob_test(n):
    p = 0
    m = 2*n
    N = n + p + 2*m
    A = np.zeros((n,p))
    G = np.eye(n)
    C = np.zeros((n,m))
    C[:,:n] = np.eye(n)
    C[:,n:] = -np.eye(n)
    b = np.zeros(p)
    d = np.array([-10]*m)
    g = np.random.rand(n)
    z = np.zeros(N)
    z[-2*m:] = 1
    return z, G, A, C, g, b, d, n, m, p

def prob_optpr1(n):
    n = 100
    m = 2*n
    p = n//2
    N = n + p + 2*m
    A = read_matrix('optpr1/A.dad', (n, p))
    G = read_matrix('optpr1/G.dad', (n, n), True)
    C = read_matrix('optpr1/C.dad', (n, m))
    b = read_vector('optpr1/b.dad', p)
    d = read_vector('optpr1/d.dad', m)
    g = read_vector('optpr1/g_small.dad', n)
    z = np.zeros(N)
    z[n:] = 1
    return z, G, A, C, g, b, d, n, m, p

def prob_optpr2(n):
    n = 1000
    m = 2*n
    p = n//2
    N = n + p + 2*m
    A = read_matrix('optpr2/A.dad', (n, p))
    G = read_matrix('optpr2/G.dad', (n, n), True)
    C = read_matrix('optpr2/C.dad', (n, m))
    b = read_vector('optpr2/b.dad', p)
    d = read_vector('optpr2/d.dad', m)
    g = read_vector('optpr2/g_small.dad', n)
    z = np.zeros(N)
    z[n:] = 1
    return z, G, A, C, g, b, d, n, m, p

### functions useful for PLOTs

def avg(list):
    return sum(list)/len(list)

solver_list = ['gepp_solver', 'ldl_solver', 'pseudo_ldl_solver', 'chol_solver', 'cho_scipy_solver']

def solver_list_picker(strategy,problem,phase=0):
    opts = [prob_optpr1, prob_optpr2]
    if strategy=='s0':
        return ['gepp_solver']
    elif strategy=='s1':
        if phase==0 and problem not in opts :
            return ['gepp_solver', 'ldl_solver', 'pseudo_ldl_solver']
        else:
            return ['gepp_solver', 'pseudo_ldl_solver']
    elif strategy=='s2':
        if phase == 0:
            return ['gepp_solver', 'ldl_solver', 'pseudo_ldl_solver', 'chol_solver', 'cho_scipy_solver']
        else:
            return ['gepp_solver', 'ldl_solver', 'pseudo_ldl_solver', 'chol_solver', 'cho_scipy_solver']

def divisors(a):
    divs =  np.array([])
    sqrta = int(a**0.5)
    print('sqrt:',sqrta)
    for i in range(1,sqrta+1):
        b = a%i
        if b == 0.:
            otheri = a//i
            divs = np.append(divs,i)
            if i != sqrta:
                divs = np.append(divs,otheri)
            print(i)
    return np.sort(divs.astype('int'))

def plot_time_iter(problem, strategy, solver_list, file, n_list, reps, lu_fixedblock_solver=[]):
    #with open(filename+'.dat','w') as file:
    print('solver_list')
    n_solvers = len(solver_list)
    file.write('#TIME VS DIMENSION \n#total idxs:{}\n'.format(n_solvers))
    for solver_name, idx in zip(solver_list, range(n_solvers)):
        print(solver_name)
        file.write('#{}({}) REPS:{}\n#n(1) t(2) t_bar(3) iter(4) iter_bar(5)\n'.format(solver_name, idx, reps))
        solver = eval(solver_name)
        def optimizator(n):
            return optimization(n, problem, strategy, solver)
        t_list, iter_list = time_iter(optimizator, reps, n_list)

        for n, t, t_bar, iter, iter_bar in zip(n_list, t_list[0], t_list[1], iter_list[0], iter_list[1]):
            file.write('{} {} {} {} {}\n'.format(n, t, t_bar, iter, iter_bar))
        file.write('\n\n')
    return

def time_iter(optimizator,reps, n_list):
    t_list = [[],[]]
    iter_list = [[],[]]
    for n in n_list:
        t_list_reps = []
        iter_list_reps = []
        for _ in range(reps):
            t, its = optimizator(n)
            t_list_reps.append(t)
            iter_list_reps.append(its)
        t_list[0].append(np.mean(t_list_reps))
        t_list[1].append(np.std(t_list_reps))
        iter_list[0].append(np.mean(iter_list_reps))
        iter_list[1].append(np.std(iter_list_reps))
    return t_list, iter_list



def plot_cond(problem, strategy, solver_list, file, n_list, lu_fixedblock_solver=[]):
    # with open(filename+'.dat','w') as file:
    n_solvers = len(solver_list)
    file.write('#CONDITION NUMBER VS ITERATIONS \n#total idxs:{}\n'.format(n_solvers))
    columns_label = '#iters(1)'
    n_range = len(n_list)
    for n, ni in zip(n_list, range(n_range)):
        columns_label = columns_label + ' n={}({})'.format(n,ni+2)

    for solver_name, idx in zip(solver_list, range(n_solvers)):
        print(solver_name)
        file.write('#{}({})\n'.format(solver_name, idx + n_solvers)+columns_label+'\n')
        solver = eval(solver_name)
        cond_nb_list = []
        print('n_list in plot COND:', n_list,' and its size:',len(n_list))
        for n in n_list:
            _, _, cond_nb =  optimization(n, problem, strategy, solver, cond_nb= True)
            print('at least one opt finished')
            cond_nb_list.append(cond_nb)

        min_iter = min([len(lista) for lista in cond_nb_list])
        #iter_list = list(range(1,miniter+1))

        for iter in range(min_iter):
            file.write('{} '.format(iter+1))
            for i in range(n_range):
                file.write('{} '.format(cond_nb_list[i][iter] ))
            file.write('\n')
        file.write('\n\n')
    return

def plot_blocks(problem, strategy, solver_list, file, n, reps):
    file.write('#BLOCK ANALSYS\n')
    solver_list, lu_fixedblock_solver = add_lu_block_list(solver_list, n)
    n_solvers = len(solver_list)
    for solver_name, idx in zip(solver_list, range(n_solvers)):
        solver = eval(solver_name)
        t_list = []
        for _ in range(reps):
            t, _ = optimization(n, problem, strategy, solver)
            t_list.append(t)
        t = np.mean(t_list)
        t_std = np.std(t_list)
        file.write('{} {} {} {}\n'.format(idx, solver_name, t, t_std))
    return


def plot_all(problem, strategy_str, filename, n_list_titer, n_list_cond, n_block, reps,lu_fixedblock_solver):
    opts = [prob_optpr1, prob_optpr2]
    solver_list = solver_list_picker(strategy_str,problem)
    print('phase0:',solver_list)
    strategy = eval(strategy_str)
    with open(filename+'.dat','w') as file:
        print('TITER:',n_list_titer)
        plot_time_iter(problem, strategy, solver_list, file, n_list_titer, reps, lu_fixedblock_solver)
        print('COND N', n_list_cond)
        plot_cond(problem, strategy, solver_list, file, n_list_cond, lu_fixedblock_solver)
        solver_list = solver_list_picker(strategy_str,1)
        print('phase1:',solver_list)
        if strategy_str != 's0' and  problem not in opts:
            plot_blocks(problem, strategy, solver_list, file, n_block, reps)
    return

def add_lu_block_list(solver_list, n):
    name_dict = 'lu_fixedblock_solver[{}]'
    lu_fixedblock_solver = {}
    if type(n)==int:
        n_list = divisors(n)
        print(n_list)
    elif type(n)==list:
        n_list = n
    else:
        print('error')
    for n in n_list:
        solver_list.append(name_dict.format(n))
        lu_fixedblock_solver[n] = gen_lu_fixedblock_solver(n)
    return solver_list, lu_fixedblock_solver