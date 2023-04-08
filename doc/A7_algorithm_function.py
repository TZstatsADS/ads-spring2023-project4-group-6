import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import itertools

df=pd.read_csv("../output/compas-scores-two-years_cleaned_A7.csv")

## information
def information(arr1,arr2):
    l=len(arr1)
    sze=len(arr1[0])
    ans=0
    s=set()
    for i in range(l):
        t=np.concatenate([arr1[i],arr2[i]])
        s.add(tuple(t))
    for perm in s:
        pab=0
        pa=0
        pb=0
        for i in range(l):
            tmp=tuple(np.concatenate([arr1[i],arr2[i]]))
            if perm==tmp:
                pab+=1
            if perm[:sze]==tmp[:sze]:
                pa+=1
            if perm[sze:]==tmp[sze:]:
                pb+=1
        pab/=l
        pa/=l
        pb/=l
        ans+=(pab*np.log(pab/(pa*pb)))
    return ans



## conditional_information
def conditional_information(arr1,arr2,condition):
    l=len(arr1)
    sze1=len(arr1[0])
    sze2=len(arr2[0])
    ans=0
    s=set()
    for i in range(l):
        t=np.concatenate([arr1[i],arr2[i]])
        t=np.concatenate([t,condition[i]])
        s.add(tuple(t))
    for perm in s:
        pabc=0
        pac=0
        pbc=0
        pc=0
        for i in range(l):
            t=np.concatenate([arr1[i],arr2[i]])
            t=np.concatenate([t,condition[i]])
            t=tuple(t)
            if perm==t:
                pabc+=1
            if perm[sze1+sze2:]==t[sze1+sze2:]:
                pc+=1
                if perm[sze1:sze1+sze2]==t[sze1:sze1+sze2]:
                    pbc+=1
                if perm[:sze1]==t[:sze1]:
                    pac+=1
        pabc/=l
        pac/=l
        pbc/=l
        pc/=l
        ans+=(pabc*(np.log((pabc*pc)/(pbc*pac))))
    return ans


##accuracy_coefficient
def accuracy_coefficient(y,x_s,x_s_c,a):
    tmp=np.concatenate((x_s_c,a),axis=1)
    return conditional_information(y,x_s,tmp)


## discrimination_coefficient
def discrimination_coefficient(y,x_s,a):
    tmp=np.concatenate((x_s,a),axis=1)
    return information(y,tmp)*information(x_s,a)*conditional_information(x_s,a,y)


## shapley_accuracy
def shapley_accuracy(removed,y,a):
    arr=["sex","age_cat","priors_count","c_charge_degree","length_of_stay"]
    n=len(arr)
    arr.remove(removed)
    ans=0
    for i in range(1,len(arr)):
        t=list(itertools.combinations(arr, i))
        c=(np.math.factorial(i)*np.math.factorial(n-i-1))/(np.math.factorial(n))
        for comb in t:
            arr1=[]
            arr2=[]
            for k in range(i):
                arr1.append(comb[k])
            for k in range(len(arr)):
                if arr[k] not in arr1:
                    arr2.append(arr[k])
            arr1.append(removed)
            val1=df[arr1[0]]
            val2=df[arr2[0]]
            val1=val1.to_numpy()
            val2=val2.to_numpy()
            val1=np.reshape(val1,(-1,1))
            val2=np.reshape(val2,(-1,1))
            for k in range(1,len(arr1)):
                tmp=df[arr1[k]]
                tmp=tmp.to_numpy()
                tmp=np.reshape(tmp,(-1,1))
                val1=np.concatenate((val1,tmp),axis=1)
            for k in range(1,len(arr2)):
                tmp=df[arr2[k]]
                tmp=tmp.to_numpy()
                tmp=np.reshape(tmp,(-1,1))
                val2=np.concatenate((val2,tmp),axis=1)
            withaccuracy=accuracy_coefficient(y,val1,val2,a)
            
            arr1.remove(removed)
            arr2.append(removed)
            val1=df[arr1[0]]
            val2=df[arr2[0]]
            val1=val1.to_numpy()
            val2=val2.to_numpy()
            val1=np.reshape(val1,(-1,1))
            val2=np.reshape(val2,(-1,1))
            for k in range(1,len(arr1)):
                tmp=df[arr1[k]]
                tmp=tmp.to_numpy()
                tmp=np.reshape(tmp,(-1,1))
                val1=np.concatenate((val1,tmp),axis=1)
            for k in range(1,len(arr2)):
                tmp=df[arr2[k]]
                tmp=tmp.to_numpy()
                tmp=np.reshape(tmp,(-1,1))
                val2=np.concatenate((val2,tmp),axis=1)
            withoutaccuracy=accuracy_coefficient(y,val1,val2,a)
            ans+=c*(withaccuracy-withoutaccuracy)
    return ans


#shapley_discrimination
def shapley_discrimination(removed,y,a):
    arr=["sex","age_cat","priors_count","c_charge_degree","length_of_stay"]
    n=len(arr)
    arr.remove(removed)
    ans=0
    for i in range(1,len(arr)):
        t=list(itertools.combinations(arr, i))
        c=(np.math.factorial(i)*np.math.factorial(n-i-1))/(np.math.factorial(n))
        for comb in t:
            arr1=[]
            arr2=[]
            for k in range(i):
                arr1.append(comb[k])
            arr1.append(removed)
            val1=df[arr1[0]]
            val1=val1.to_numpy()
            val1=np.reshape(val1,(-1,1))
            for k in range(1,len(arr1)):
                tmp=df[arr1[k]]
                tmp=tmp.to_numpy()
                tmp=np.reshape(tmp,(-1,1))
                val1=np.concatenate((val1,tmp),axis=1)
            withaccuracy=discrimination_coefficient(y,val1,a)
            
            arr1.remove(removed)
            val1=df[arr1[0]]
            val1=val1.to_numpy()
            val1=np.reshape(val1,(-1,1))
            for k in range(1,len(arr1)):
                tmp=df[arr1[k]]
                tmp=tmp.to_numpy()
                tmp=np.reshape(tmp,(-1,1))
                val1=np.concatenate((val1,tmp),axis=1)
            withoutaccuracy=discrimination_coefficient(y,val1,a)
            ans+=c*(withaccuracy-withoutaccuracy)
    return ans