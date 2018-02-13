import numpy as np

def DE(eval, _gen_max, _max, _min, _NP, _D):
    gen_max = _gen_max
    max = _max
    min = _min
    NP = _NP    # number of vectors
    D = _D    # number of parameters
    x1 = np.zeros((NP, D))
    x2 = np.zeros((NP, D))
    trail = np.zeros((1, D))
    one = np.ones((1, D))
    cost = np.zeros(NP)
    l = 1
    m = 1
    n = 1
    CR = 0.5
    F = 1    #from 0 to 2
    
    for i in range(0, NP):
#         np.random.seed(100)
        x1[i][:] = np.random.rand(1, D) * (max - min) + min * one
        print (i)
        print (x1[i])
        f = open('output.txt', 'a')
        print (i, file = f)
        print (x1[i], file = f)
        f.close()
        w = x1[i]
        cost[i] = 0
        cost[i] = eval(w)
        
    for count in range(0, gen_max):
        for i in range(0, NP):
            
            while (l == i):
                l = np.random.randint(NP)
            while (m == i or m == l):
                m = np.random.randint(NP)
            while (n == i or n == l or n == m):
                n = np.random.randint(NP)
            
            j = np.random.randint(D)
            
            for k in range(0, D):
                if (np.random.rand(1)[0] < CR or k == D-1):
                    trail[0][j] = x1[n][j] + F * (x1[l][j] - x1[m][j])
                    if (trail[0][j] < min):
                        trail[0][j] = x1[i][j]
                    if (trail[0][j] > max):
                        trail[0][j] = x1[i][j]
                else:
                    trail[0][j] = x1[i][j]
                j = (j+1) % D

            score = eval(trail[0])
            
            if (score <= cost[i]):
                for j in range(0,D):
                    x2[i][j] = trail[0][j]
                    cost[i] = score
            else:
                for j in range(0,D):
                    x2[i][j] = x1[i][j]
            
        for i in range(0, NP):
            for j in range(0, D):
                x1[i][j] = x2[i][j]
        
        best_para = x1[0]
        best_cost = cost[0]
        for i in range(0, NP):
            if (cost[i] < best_cost):
                best_cost = cost[i]
                best_para = x1[i]
        print ("Round:  ",  count)
        print ("best_cost:  ", best_cost)
        print ("best_para:  ", best_para)
        f = open('output.txt', 'a')
        print ("Round:  ",  count, file = f)
        print ("best_cost:  ", best_cost, file = f)
        print ("best_para:  ", best_para, file = f)
        f.close()
        for i in range(0, NP):
            print ("Set:   ",  i)
            print ("para:  ", x1[i])        
            print ("Cost:  ", cost[i])
            f = open('output.txt', 'a')
            print ("Set:   ",  i, file = f)
            print ("para:  ", x1[i], file = f)        
            print ("Cost:  ", cost[i], file = f)
            f.close()

    print ("Final Result:")
    f = open('output.txt', 'a')
    print ("Final Result:", file = f)
    f.close()
    for i in range(0, NP):
        print ("Set:   ",  i)
        print ("para:  ", x1[i])        
        print ("Cost:  ", cost[i])
        f = open('output.txt', 'a')
        print ("Set:   ",  i, file = f)
        print ("para:  ", x1[i], file = f)        
        print ("Cost:  ", cost[i], file = f)
        f.close()

    return best_cost
