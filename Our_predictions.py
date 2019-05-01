# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 03:09:30 2016

@author: lanndy
(robert sep 22 edit: added one column, total number of races run)
"""

import csv
import numpy as np
data=[]
csvfile = file('Project1_data.csv', 'rb')
reade = csv.reader(csvfile)
for line in reade:
    data.append(line)
data[769][5]='F50-59'#This guy has no age information, i will give him or her one.#
origdata=data
for person in data:
    i=3
    while i<len(person):
        if person[i] in ['Marathon','Ottawa Marathon','Full Marathon','AMJ Capbell Full Marathon','Scotiabank Full Marathon']:
            i=i+5
        else:
            del person[i-2:i+3]
attend=[]
logistic_data=[0,0,0,0]
del data[0]

for person in data :
    
    i=1   
    a=0
    b=0
    c=0
    d=0
    e=0

    person1=[]
    while i<len(person):
        if person[i]=='2012-09-23':
            a=1
        elif person[i]=='2013-09-22':
            b=1
        elif person[i]=='2014-09-28':
            c=1
        elif person[i]=='2015-09-20':
            d=1
        if person[i+4][0]=='M':
           e=1
        elif person[i+4][0]=='F':
            e=0
        age=person[i+4][1:3]
        if age=='0-':
            age='50'
        elif age=='O ':
            age='50'
        i=i+5
    person1.append(person[0])
    person1.append(a)
    person1.append(b)
    person1.append(c)
    person1.append(d)
    person1.append(e)
    person1.append(int(age))
    person1.append(i/5) # i/5 = number of races run
    attend.append(person1)
    a1=np.array(person1[1:5])
    logistic_data=np.vstack((logistic_data,a1))
    logisticdata=logistic_data[1:]
    
    
csvfile = file('csv_test.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerows(attend)
csvfile.close()
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ROBERT'S CODE STARTS HERE - LOGISTIC REGRESSION GRADIENT DESCENT
import csv
import numpy as np
import math

csvfile = file('csv_test.csv', 'rb')
data = []
reade = csv.reader(csvfile)
for line in reade:
    line = [int(i) for i in line]
    data.append(line)

def sigma(a):
    return (1/(1+math.exp(-1*a)))

def norm(v): #norm, used to calculate stopping condition norm(w-w_new)>epsilon
    norm_v = 0
    for vi in v:
        norm_v = norm_v + vi**2
    norm_v = norm_v**.5
    return norm_v

def error(wo): #Error function, Err(w) in notes (over whole data). sanity check: error should decrease every step
    error_total=0
    for jo in range(len(data)):
        xio = np.array([1] + data[jo][1:4] + [data[jo][5]] + [data[jo][7]])
        yio = data[jo][4] #number
        error_total = error_total + yio*math.log(sigma(wo.dot(xio)))+(1-yio)*math.log(1-sigma(wo.dot(xio)))
    return -1*error_total

def percent(z, data_range): #percent correct on whole training data
    count=0
    for i in data_range:
        x_i = np.array([1] + data[i][1:4] + [data[i][5]] + [data[i][7]])
        y_i = data[i][4]
        if ((sigma(z.dot(x_i))>.5 and y_i==1) or (sigma(z.dot(x_i))<=.5 and y_i==0)):
            count = count+1
    percento=float(count)/len(data_range)
    return percento


# GRADIENT DESCENT
def gradient_descent(w_new, data_range):
    #initialize 
    w = np.array([0,0,0,0,0,0])
    xi = np.array([0,0,0,0,0,0])
    sum_this=np.array([0,0,0,0,0,0])
    iteration=0
    
    #constants to choose
    alpha = .001 #step factor
    epsilon = .001 #stop condition
    
    #for k in range(100):          #fixed number of iterations for w_k -> w_k+1
    while norm(w-w_new)>epsilon:   #stop at epsilon
        iteration = iteration + 1
        w = w_new
        for j in data_range:
            xi = np.array([1] + data[j][1:4] + [data[j][5]] + [data[j][7]]) #basis,2012,13,14 attendance, M/F, num. races
            yi = data[j][4] # 2015 attendance
            sum_this = sum_this + xi*(yi-sigma(w.dot(xi)))
        w_new = w + sum_this*(alpha/(iteration+1))
        #print norm(w-w_new)
    print (w)
    return(w)


#4-fold validation
w_initial = np.array([-1.2,-6.9,-5.8,-6.3,2.0,1.8]) # our "guess" for initial w
nums = range(len(data))

w1 = gradient_descent(w_initial, nums[len(data)/4 : len(data)])                           #test data is first 1/4
w1_error = percent(w1,nums[0 : len(data)/4])
print w1_error

w2 = gradient_descent(w_initial, nums[0 : len(data)/4] + nums[len(data)/2 : len(data)])   # 2nd 1/4
w2_error = percent(w2,nums[len(data)/4 : len(data)/2])
print w2_error

w3 = gradient_descent(w_initial, nums[0 : len(data)/2] + nums[len(data)*3/4 : len(data)]) # 3rd 1/4
w3_error = percent(w3,nums[len(data)/2 : len(data)*3/4])
print w3_error

w4 = gradient_descent(w_initial, nums[0 : len(data)*3/4])                                 # 4th 1/4
w4_error = percent(w4,nums[len(data)*3/4: len(data)])
print w4_error

print (w1_error + w2_error + w3_error + w4_error)/4 #average error
w_final = (w1+w2+w3+w4)/4 #average weights
print w_final

countt=0
data_predictions= []
for i in range(len(data)):
    #data_predictions[i]=
    xi = np.array([1] + data[i][2:6] + [data[i][7]]) #basis, 2013-2015 race attendence, M/F, number of races done
    prob = sigma(w_final.dot(xi))
    if prob>.5:
        data_predictions.append(1)
        countt=countt+1
    else:
        data_predictions.append(0)
print(countt)
data_predictions.insert(0,'Y1_LOGISTIC')

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 03:09:30 2016

@author: lanndy
"""

import csv
import numpy as np
from collections import Counter  
data=[]
csvfile = file('Project1_data.csv', 'rb')
reade = csv.reader(csvfile)
for line in reade:
    data.append(line)
origdata=data
for person in data:
    i=3
    while i<len(person):
        if person[i] in ['Marathon','Ottawa Marathon','Full Marathon','AMJ Capbell Full Marathon','Scotiabank Full Marathon']:
            i=i+5
        else:
            del person[i-2:i+3]
attend=[]
logistic_data=[0,0,0,0]
del data[0]
data[768][5]='F50-59'#This guy has no age information, i will give him or her one.
for person in data :
    i=1   
    a=0
    b=0
    c=0
    d=0
    e=0

    person1=[]
    while i<len(person):
        if person[i]=='2012-09-23':
            a=1
        elif person[i]=='2013-09-22':
            b=1
        elif person[i]=='2014-09-28':
            c=1
        elif person[i]=='2015-09-20':
            d=1
        if person[i+4][0]=='M':
            e=1
        elif person[i+4][0]=='F':
            e=0
        age=person[i+4][1:3]
        if age=='0-':
            age='50'
        elif age=='O ':
            age='50'
        elif age=='12':
            age='15'
        elif age=='19':
            age='18'
        elif age=='20':
            age='18'
        elif age=='24':
            age='25'
        i=i+5
    person1.append(int(person[0]))
    person1.append(a)
    person1.append(b)
    person1.append(c)
    person1.append(d)
    person1.append(e)
    person1.append(int(age))
    attend.append(person1)
    a1=np.array(person1[1:5])
    logistic_data=np.vstack((logistic_data,a1))
    logisticdata=logistic_data[1:]

#P(come or not)    
naive_data=np.array(attend)
p_2015come=float(sum(naive_data[:,4]))/float(len(naive_data[:,4]))
p_2015notcome=1-p_2015come

def funcp_feature_givencom(person,p_2012come_given2015come,p_2013come_given2015come,p_2014come_given2015come,p_male_given2015come,p_age_2015come):

    a=1
    if person[1]==1:
        a=a*p_2012come_given2015come
    else:
        a=a*(1-p_2012come_given2015come)
    if person[2]==1:
        a=a*p_2013come_given2015come
    else:
        a=a*(1-p_2013come_given2015come)
    if person[3]==1:
        a=a*p_2014come_given2015come
    else:
        a=a*(1-p_2014come_given2015come)
    if person[5]==1:
        a=a*p_male_given2015come
    else:
        a=a*(1-p_male_given2015come)
    a=a*p_age_2015come[person[6]]
    return(a)

def funcp_feature_givennotcom(person,p_2012come_given2015notcome,p_2013come_given2015notcome,p_2014come_given2015notcome,p_male_given2015notcome,p_age_2015notcome):

    a=1
    if person[1]==1:
        a=a*p_2012come_given2015notcome
    else:
        a=a*(1-p_2012come_given2015notcome)
    if person[2]==1:
        a=a*p_2013come_given2015notcome
    else:
        a=a*(1-p_2013come_given2015notcome)
    if person[3]==1:
        a=a*p_2014come_given2015notcome
    else:
        a=a*(1-p_2014come_given2015notcome)
    if person[5]==1:
        a=a*p_male_given2015notcome
    else:
        a=a*(1-p_male_given2015notcome)
    a=a*p_age_2015notcome[person[6]]
    return(a)
#cross-validation

attend_train=attend[0:len(attend)/2]
attend_validation=attend[len(attend)/2:len(attend)]
#this is the second fold cross-validation
#attend_validation=attend[0:len(attend)/2]
#attend_train=attend[len(attend)/2:len(attend)]
attend_2015come=[]
attend_2015notcome=[]
for person in attend_train:
    if person[4]==1:
        attend_2015come.append(person)
for person in attend_train:
    if person[4]==0:
        attend_2015notcome.append(person)
attend_2015come=np.array(attend_2015come)
attend_2015notcome=np.array(attend_2015notcome)
p_2012come_given2015come=float(sum(attend_2015come[:,1]))/float(len(attend_2015come))
p_2013come_given2015come=float(sum(attend_2015come[:,2]))/float(len(attend_2015come))
p_2014come_given2015come=float(sum(attend_2015come[:,3]))/float(len(attend_2015come))
p_male_given2015come=float(sum(attend_2015come[:,5]))/float(len(attend_2015come))
p_2012come_given2015notcome=float(sum(attend_2015notcome[:,1]))/float(len(attend_2015notcome))
p_2013come_given2015notcome=float(sum(attend_2015notcome[:,2]))/float(len(attend_2015notcome))
p_2014come_given2015notcome=float(sum(attend_2015notcome[:,3]))/float(len(attend_2015notcome))
p_male_given2015notcome=float(sum(attend_2015notcome[:,5]))/float(len(attend_2015notcome))
p_age_2015come=Counter(attend_2015come[:,6])
p_age_2015notcome=Counter(attend_2015notcome[:,6])
for key in p_age_2015come.keys():
    p_age_2015come[key]=float(p_age_2015come[key])/float(len(attend_2015come))
for key in p_age_2015notcome.keys():
    p_age_2015notcome[key]=float(p_age_2015notcome[key])/float(len(attend_2015notcome))
    
predict_naive1=[]
predict_naive0=[]
predict_naive=[]

for person in attend_validation:
    p_feature_given2015come=funcp_feature_givencom(person,p_2012come_given2015come,p_2013come_given2015come,p_2014come_given2015come,p_male_given2015come,p_age_2015come)
    p_feature_given2015notcome=funcp_feature_givennotcom(person,p_2012come_given2015notcome,p_2013come_given2015notcome,p_2014come_given2015notcome,p_male_given2015notcome,p_age_2015notcome)
    predict_naive0.append(p_feature_given2015notcome)
    predict_naive1.append(p_feature_given2015come)
    if p_feature_given2015come>p_feature_given2015notcome:
        predict_naive.append(1)
    else:
        predict_naive.append(0)
naive_data_validation=np.array(attend_validation)
v = list(map(lambda x: x[0]-x[1], zip(naive_data_validation[:,4], predict_naive))) 
a=Counter(v)
accuracy_validation=a[0]/float(len(naive_data_validation[:,4]))

for person in attend_train:
    p_feature_given2015come=funcp_feature_givencom(person,p_2012come_given2015come,p_2013come_given2015come,p_2014come_given2015come,p_male_given2015come,p_age_2015come)
    p_feature_given2015notcome=funcp_feature_givennotcom(person,p_2012come_given2015notcome,p_2013come_given2015notcome,p_2014come_given2015notcome,p_male_given2015notcome,p_age_2015notcome)
    predict_naive0.append(p_feature_given2015notcome)
    predict_naive1.append(p_feature_given2015come)
    if p_feature_given2015come>p_feature_given2015notcome:
        predict_naive.append(1)
    else:
        predict_naive.append(0)
naive_data_train=np.array(attend_train)
v = list(map(lambda x: x[0]-x[1], zip(naive_data_validation[:,4], predict_naive))) 
a=Counter(v)
accuracy_train=a[0]/float(len(naive_data_train[:,4]))

#P(feature|come or not)
attend_2015come=[]
attend_2015notcome=[]
for person in attend:
    if person[4]==1:
        attend_2015come.append(person)
for person in attend:
    if person[4]==0:
        attend_2015notcome.append(person)
attend_2015come=np.array(attend_2015come)
attend_2015notcome=np.array(attend_2015notcome)
p_2012come_given2015come=float(sum(attend_2015come[:,1]))/float(len(attend_2015come))
p_2013come_given2015come=float(sum(attend_2015come[:,2]))/float(len(attend_2015come))
p_2014come_given2015come=float(sum(attend_2015come[:,3]))/float(len(attend_2015come))
p_male_given2015come=float(sum(attend_2015come[:,5]))/float(len(attend_2015come))
p_2012come_given2015notcome=float(sum(attend_2015notcome[:,1]))/float(len(attend_2015notcome))
p_2013come_given2015notcome=float(sum(attend_2015notcome[:,2]))/float(len(attend_2015notcome))
p_2014come_given2015notcome=float(sum(attend_2015notcome[:,3]))/float(len(attend_2015notcome))
p_male_given2015notcome=float(sum(attend_2015notcome[:,5]))/float(len(attend_2015notcome))
p_age_2015come=Counter(attend_2015come[:,6])
p_age_2015notcome=Counter(attend_2015notcome[:,6])
for key in p_age_2015come.keys():
    p_age_2015come[key]=float(p_age_2015come[key])/float(len(attend_2015come))
for key in p_age_2015notcome.keys():
    p_age_2015notcome[key]=float(p_age_2015notcome[key])/float(len(attend_2015notcome))
    
predict_naive1=[]
predict_naive0=[]
predict_naive=[]

for person in attend:
    p_feature_given2015come=funcp_feature_givencom(person,p_2012come_given2015come,p_2013come_given2015come,p_2014come_given2015come,p_male_given2015come,p_age_2015come)
    p_feature_given2015notcome=funcp_feature_givennotcom(person,p_2012come_given2015notcome,p_2013come_given2015notcome,p_2014come_given2015notcome,p_male_given2015notcome,p_age_2015notcome)
    predict_naive0.append(p_feature_given2015notcome)
    predict_naive1.append(p_feature_given2015come)
    if p_feature_given2015come>p_feature_given2015notcome:
        predict_naive.append(1)
    else:
        predict_naive.append(0)
        
v = list(map(lambda x: x[0]-x[1], zip(naive_data[:,4], predict_naive))) 
a=Counter(v)
accuracy=a[0]/float(len(naive_data[:,4]))

#predicting 2016
attend_new=attend
predict2016_naive=[]
predict2016_naive0=[]
predict2016_naive1=[]
for person in attend_new:
    person[1]=person[2]
    person[2]=person[3]
    person[3]=person[4]
for person in attend_new:
    p_feature_given2015come=funcp_feature_givencom(person,p_2012come_given2015come,p_2013come_given2015come,p_2014come_given2015come,p_male_given2015come,p_age_2015come)
    p_feature_given2015notcome=funcp_feature_givennotcom(person,p_2012come_given2015notcome/7,p_2013come_given2015notcome/8,p_2014come_given2015notcome/8,p_male_given2015notcome,p_age_2015notcome)
    predict2016_naive0.append(p_feature_given2015notcome)
    predict2016_naive1.append(p_feature_given2015come)
    if person[1]==0 and person[2]==0 and person[3]==0:
        predict2016_naive.append(0)
    elif p_feature_given2015come>p_feature_given2015notcome:
        predict2016_naive.append(1)
    else:
        predict2016_naive.append(0)
predict2016_naive.insert(0,'Y1_NAIVEBAYES')    
predict2016_naive=list(zip(predict2016_naive,data_predictions)) #OUR PREDICTION LISTS GO HERE (plus one title line)

csvfile=file('csv_test.csv', 'wb')
writer=csv.writer(csvfile)
writer.writerows(predict2016_naive)
csvfile.close()