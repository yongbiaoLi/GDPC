import numpy as np
import perpy as py
from perpy import clustering as cl

'''
    Function: grouping
    Parameters: 
        x: data,
        dc: cutoff distance
    Return:
        g: groups
'''
def grouping(x, dc):
    num_x = x.shape[0]  # the number of samples
    g = []  # create set g

    for i in range(num_x):  # each sample x[i]
        flag = 0  # 0: create new group; 1: not
        for j in range(len(g)):  # each group g[j]
            if py.dist(x[i], x[g[j][0]]) < dc:  # the distance of bewteen sample x[i] and group g[j] is less than dc
                g[j].append(i)  # add smaple x[i] to group g[j]
                flag = 1
                break  # next sample
        if flag == 0:  # create new group
            g.append([i])

    return g


'''
    Function:
        compute group density and determine core of boundary group
    Parameters:
        g: groups
    return:
        core: core groups
        boundary: boundary groups
'''

'''
    Function:
        compute group density and determine core of boundary group
    Parameters:
        g: groups
    return:
        core: core groups
        boundary: boundary groups
'''


def density(g):
    num_g = len(g)  # the number of group
    rho = np.zeros(num_g)  # density

    for i in range(num_g):  # compute density of each group
        rho[i] = len(g[i]) - 1

    sort_rho = np.sort(rho)[::-1]  # density sort
    sort_rho = sort_rho[np.nonzero(sort_rho)]  # non zero density

    index = np.ceil(0.7 * len(sort_rho)) - 1  # threshold index, 70%
    threshold = sort_rho[int(index)]  # threshold

    sort_g = np.argsort(-rho)  #
    core = []
    boundary = []

    # determine core or boundary group
    for i in range(num_g):  # each group
        if rho[sort_g[i]] >= threshold:
            core.append(sort_g[i])
        else:
            boundary.append(sort_g[i])

    return core, boundary


'''
    Function:
        determine labels of sample.
    Parameters:
        x: data,
        g: group,
        core: core set,
        boundary: boundary set,
        dc: cutoff distance.
    Return:
        r: labels,
        t: cluster centers.
'''


def clustering(x, g, core, boundary, dc):
    c = []  # cluster
    t = []  # cluster centers

    # determine labels of core group
    while len(core) != 0:
        flag = 0

        for j in range(len(core)):
            if j >= len(core):
                break
            # each sample in the each group
            for k in range(len(c)):
                for i in range(len(c[k])):
                    if py.dist(x[g[core[j]][0]], x[c[k]][i]) <= 2 * dc:
                        c[k] = c[k] + g[core[j]]
                        core.remove(core[j])
                        flag = 1
                        break
                if flag == 1:
                    break

        # create new cluster
        if flag == 0:
            c.append(g[core[0]])  # new cluster
            t.append(g[core[0]][0])  # new cluster center
            core.remove(core[0])

    r = [len(c) for i in range(len(x))]  # labels

    for i in range(len(c)):
        for j in range(len(c[i])):
            r[c[i][j]] = i

    temp_r = r
    # determine labels of boundary group
    for i in range(len(boundary)):
        min_dist = 999
        nest_c = 999
        flag = 0

        for j in range(len(temp_r)):
            if temp_r[j] == len(c):
                continue
            if py.dist(x[j], x[g[boundary[i]][0]]) <= 2 * dc:
                for b in g[boundary[i]]:
                    r[b] = r[j]
                flag = 1
                break

        if flag == 1:
            continue

        for k in range(len(t)):
            if py.dist(x[t[k]], x[g[boundary[i]][0]]) < min_dist:  # nest cluster
                min_dist = py.dist(x[t[k]], x[g[boundary[i]][0]])
                nest_c = k
        r[g[boundary[i]][0]] = r[nest_c]

    return r, t

if __name__ == '__main__':
    x, r1 = py.load(path='D:/待测数据集', col_labels=2)
    dc = 0.49
    g = grouping(x, dc)
    core, boundary = density(g)
    r, t = clustering(x, g, core, boundary, dc)
    # py.plt_scatter(x, r)
    print(cl.ARI(r1, r), cl.NMI(r1, r), cl.Homogeneity(r1, r))