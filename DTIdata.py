import numpy as np
from math import sqrt
import xlrd
import os
import scipy.stats as stats

def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab

def f(x):
    return x * x

def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    if den == 0:
        num = 0
    else:
        num = num / den
    return num

def load_dti_graph(path='F:/qyl/MSGTN/dataset/DTI/', mcitype=None):
    if mcitype == 'LMCI/':
        objects = np.zeros([38, 90, 90])  # loda dti data LMCI
    else:
        objects = np.zeros([44, 90, 90])
    pathDir = os.listdir("{}{}".format(path, mcitype))  # Specified folder path
    i = 0
    for subject in pathDir:
        objects[i] = np.genfromtxt("{}{}{}.".format(path, mcitype, subject))
        i = i + 1  # objects.shape(38,90,90)
    return objects  # load_dti_data

def load_fmri_data(path='F:/qyl/MSGTN/dataset/fMRI/', mcitype=None):
    # if mcitype == 'NC/':
    #     objects = np.zeros([44, 170, 90])
    #     labels = np.zeros([44, 2])
    #     for i in range(44):
    #         labels[i, 0] = 0
    if mcitype == 'EMCI/':
        objects = np.zeros([44, 170, 90])
        labels = np.zeros([44, 2])
        for i in range(44):
            labels[i, 1] = 1
    if mcitype == 'SMC/':
        objects = np.zeros([44, 170, 90])
        labels = np.zeros([44, 2])
        for i in range(44):
            labels[i, 0] = 0
    # if mcitype == 'LMCI/':
    #     labels = np.zeros([38, 2])
    #     objects = np.zeros([38, 170, 90])
    #     for i in range(38):
    #         labels[i, 1] = 1  # label setup, classify about NC / LMCI
    pathDir = os.listdir("{}{}".format(path, mcitype))
    i = 0
    for subject in pathDir:
        objects[i] = np.genfromtxt("{}{}{}.".format(path, mcitype, subject))
        i = i + 1
    objects = objects.transpose((0, 2, 1))  # mcitype,subject
    return objects, labels

# return fmri,dti,labels
# Vstack();fmri dti 两者不同标签融合
def construct_data():
    # fLMCI, fLlabel = load_fmri_data(mcitype='LMCI/')
    # fNC, fNlabel = load_fmri_data(mcitype='NC/')
    fEMCI, fElabel = load_fmri_data(mcitype='EMCI/')
    fSMC, fSlabel = load_fmri_data(mcitype='SMC/')
    EMCI_DTI = load_dti_graph(mcitype='EMCI/')  # label same as above
    SMC_DTI = load_dti_graph(mcitype='SMC/')
    # LMCI_DTI = load_dti_graph(mcitype='LMCI/')  # label same as above
    # NC_DTI = load_dti_graph(mcitype='NC/')
    # fmri = np.vstack((fNC, fLMCI)) # NC LMCI
    # dti = np.vstack((NC_DTI, LMCI_DTI))
    # fmri = np.vstack((fSMC, fLMCI)) # SMC LMCI
    # dti = np.vstack((SMC_DTI, LMCI_DTI))
    # fmri = np.vstack((fEMCI, fLMCI))#EMCI LMCI
    # dti = np.vstack((EMCI_DTI, LMCI_DTI))
    # fmri = np.vstack((fNC, fEMCI)) # NC EMCI
    # dti = np.vstack((NC_DTI, EMCI_DTI))
    # fmri = np.vstack((fNC, fLMCI)) #NC SMC
    # dti = np.vstack((NC_DTI, LMCI_DTI))
    fmri = np.vstack((fSMC, fEMCI)) # SMCI EMCI
    dti = np.vstack((SMC_DTI, EMCI_DTI))
    labels = np.vstack((fSlabel, fElabel))  # share labels # print(labels.shape)
    return fmri, dti, labels

def bulid_unimage():
    data = xlrd.open_workbook('dataset/subject data.xlsx')
    sex = 3
    age = 4
    invalid = [0, -1, -1]
    # NC = data.sheet_by_index(0)
    SMC = data.sheet_by_index(1)
    EMCI = data.sheet_by_index(2)
    # LMC = data.sheet_by_index(3)
    # NC_sex = NC.col_values(sex)
    # NC_age = NC.col_values(age)
    # LMC_sex = LMC.col_values(sex)
    # LMC_age = LMC.col_values(age)
    SMC_sex = SMC.col_values(sex)
    SMC_age = SMC.col_values(age)
    EMCI_sex = EMCI.col_values(sex)
    EMCI_age = EMCI.col_values(age)
    for i in invalid:
        # NC_age.pop(i)
        # NC_sex.pop(i)
        SMC_age.pop(i)
        SMC_sex.pop(i)
        EMCI_age.pop(i)
        EMCI_sex.pop(i)
        # LMC_age.pop(i)  # 38
        # LMC_sex.pop(i)


    data_age = np.zeros([1, 88])  # 44 44 38
    data_sex = np.zeros([1, 88], dtype=np.str_)
    for i in range(44):
        # data_age[0][i] = NC_age[i]
        # data_sex[0][i] = NC_sex[i]
        data_age[0][i] = SMC_age[i]
        data_sex[0][i] = SMC_sex[i]
        # data_age[0][i] = EMCI_age[i]
        # data_sex[0][i] = EMCI_sex[i]
    #  NC // VS LMCI
    # for i in range(38):
    #     data_age[0][i + 44] = LMC_age[i]
    #     data_sex[0][i + 44] = LMC_sex[i]
    for i in range(44):
        data_age[0][i + 44] = EMCI_age[i]
        data_sex[0][i + 44] = EMCI_sex[i]
    #     data_age[0][i + 44] = SMC_age[i]
    #     data_sex[0][i + 44] = SMC_sex[i]
    # adj = np.zeros([82, 82])
    adj = np.zeros([88, 88])
    for i in range(88):
        for j in range(88):
    # for i in range(82):
    #     for j in range(82):
            if data_sex[0][i] == data_sex[0][j]:
                if abs(data_age[0][i] - data_age[0][j]) == 0:
                    adj[i][j] = 1
                else:
                    adj[i][j] = 0.9 / sqrt(abs((data_age[0][i] - data_age[0][j])))
            else:
                if abs(data_age[0][i] - data_age[0][j]) == 0:
                    adj[i][j] = -1
                else:
                    adj[i][j] = -0.9 / sqrt(abs((data_age[0][i] - data_age[0][j])))

    return adj

def bulid_fcn(data):
    # ### NC // LMCI####
    # fcn = np.zeros([82, 28, 90, 90])
    # fcn1 = np.zeros((82, 90, 90))
    #### WITHOUT LMCI####
    fcn = np.zeros([88, 28, 90, 90])
    fcn1 = np.zeros((88, 90, 90))
    for s in range(88):
    # for s in range(82):
        for k in range(28):
            for i in range(90):

                # fcn[s][k][][:j] =0
                for j in range(90):

                    # print(data[s:s+1,5*i:45+i*5,:].shape)
                    # fcn[s][k][i][j] = corrcoef(data[s][i][5*k:45+k*5], data[s][j][5*k:45+k*5])
                    # fcn[s][k][i][j] = corrcoef(data[s][i][5 * k:40 + k * 5], data[s][j][5 * k:40 + k * 5])
                    fcn[s][k][i][j] = corrcoef(data[s][i][5 * k:35 + k * 5], data[s][j][5 * k:35 + k * 5])
            fcn[s][k] = np.dot(fcn[s][k], fcn[s][k])
            fcn1[s] += fcn[s][k]
        fcn1[s] = fcn1[s] / 28
        # fcn1[s] = np.dot(fcn1[s], LN_DTI[s])
        # fcn1[s] = np.dot(fcn1[s], SL_DTI[s])
        # fcn1[s] = np.dot(fcn1[s], EL_DTI[s])
        # fcn1[s] = np.dot(fcn1[s], NE_DTI[s])
        # fcn1[s] = np.dot(fcn1[s], NS_DTI[s])
        fcn1[s] = np.dot(fcn1[s], SE_DTI[s])

    return fcn1


def extra_fea(fcn):
    fea = np.zeros([fcn.shape[0], fcn.shape[1]])
    for i in range(fcn.shape[0]):
        for j in range(fcn.shape[1]):
            a = 0
            for k in range(fcn.shape[1]):
                b = fcn[i][k][j] * fcn[i][j][j] * fcn[i][j][k]
                a = (a + 2 * pow(abs(b), 1/3))
            if fcn.shape[0] * fcn.shape[1] ==0:
                a = 0
            else:
                a = a / (fcn.shape[0] * fcn.shape[1])
            fea[i][j] = a
    return fea

def bulid_graph(fea, unimage):
    u = fea.shape[0]
    v = fea.shape[1]
    adj = np.zeros([fea.shape[0], fea.shape[0]])
    b = np.zeros([u, v])
    c = np.zeros([u])
    for i in range(u):
        a = 0
        d = 0
        for j in range(v):
            a = a + fea[i][j]
        a = a / 90
        fea[i] = fea[i] - a
        b[i] = list(map(f, list(fea[i])))
        for p in range(v):
            d = d + b[i][p]
        c[i] = sqrt(d)
    for k in range(u):
        for s in range(u):
            if c[k] * c[s] == 0:
                adj[k][s] = 0
            else:
                adj[k][s] = np.dot(fea[k], fea[s]) / (c[k] * c[s])
    unimage = np.array(unimage)
    adj = adj * unimage
    return adj

# 打破顺序
permutation = list(np.random.permutation(88))  # 82、84
# permutation = list(np.random.permutation(82))
# #NL
# fMRI, DTI, Labels = construct_data()
# LN_fMRI = fMRI[permutation, :]
# LN_Labels = Labels[permutation, :]
# LN_DTI = DTI[permutation, :]
# SL
fMRI, DTI, Labels = construct_data()
SE_fMRI = fMRI[permutation, :]
SE_Labels = Labels[permutation, :]
SE_DTI = DTI[permutation, :]

np.save("need/SE_dti_draph.npy", SE_DTI)
np.save('need/SE_label.npy', SE_Labels)
# print(LN_fMRI)

###constract different
SE_fMRI_FCN = bulid_fcn(SE_fMRI)
# print(LN_fMRI_FCN)
print("Fcn network Construction successful ! !")

SE_fMRI_fea = extra_fea(SE_fMRI_FCN)
np.savetxt('need/SE_fuse_fea.txt', SE_fMRI_fea)

unimage = np.array(bulid_unimage())
adj = bulid_graph(SE_fMRI_fea, unimage)
print("Unimage Construction successful ! !")
np.savetxt('need/SE_fuse_graph.txt', adj)
