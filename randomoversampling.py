from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier    
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import collections

a1=0

row1=np.arange(1620)
results= pd.DataFrame(data=None,index=row1,columns = ['ratio', 'Class','Training Datasize','After Sampling','Precision','Recall','f1-score','oobscore','Predicted Datasize','Testing Datasize'])

data_train = pd.read_csv("train_data.csv")
mapping = {'icmp': 1, 'udp': 2,'tcp' : 3}
data_train=data_train.replace({'protocol_type': mapping})


l1=[]
for i in np.arange(len(data_train)):
    l1.append(0)
data_train['Class']=l1

data_train.loc[data_train['xAttack']=='u2r', 'Class'] = 2

#data_train=data_train[data_train.xAttack!='u2r']
#data_train=data_train[data_train.xAttack!='dos']
#data_train=data_train[data_train.xAttack!='probe']
#data_train=data_train[data_train.xAttack!='r2l']
#data_train=data_train[data_train.xAttack!='normal']


y_train1 = data_train.loc[:, data_train.columns == 'xAttack']
print("train data size",collections.Counter(y_train1['xAttack']))
traindatasize=collections.Counter(y_train1['xAttack'])
mapping1 = {'normal': 1, 'dos': 2,'probe' : 3,'r2l':4,'u2r':5}
data_train=data_train.replace({'xAttack': mapping1})
noclass_train1 = data_train.loc[:, data_train.columns != 'Class']
class_train1 = data_train.loc[:, data_train.columns == 'Class']







data_test = pd.read_csv("test_data.csv")
data_test=data_test.replace({'protocol_type': mapping})
#data_test=data_test[data_test.xAttack!='u2r']
#data_test=data_test[data_test.xAttack!='dos']
#data_test=data_test[data_test.xAttack!='probe']
#data_test=data_test[data_test.xAttack!='r2l']
#data_test=data_test[data_test.xAttack!='normal']

X_test = data_test.loc[:, data_test.columns != 'xAttack']
y_test = data_test.loc[:, data_test.columns == 'xAttack']
print("test data size",collections.Counter(y_test['xAttack']))
testdatasize=collections.Counter(y_test['xAttack'])

print("original training data size",testdatasize)




A=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
B=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
#B=[0.8]
#A=[0.8]
numofmaj1=traindatasize['normal']+traindatasize['dos']+traindatasize['probe']+traindatasize['r2l']
numofmin1=traindatasize['u2r']
noclass_train1_cols=noclass_train1.columns

for i1 in A:
    rat1=numofmin1/(numofmaj1*i1)
    print("---------------")
    print("ratio",i1)
    results['ratio'][a1]=i1
    
    if rat1<=1:
        ros1 = RandomOverSampler(sampling_strategy=rat1,random_state=5)
        
        noclass_train_sampled1,class_train_sampled1  = ros1.fit_sample(noclass_train1, class_train1.values.ravel())
        noclass_train_sampled1= pd.DataFrame(data=noclass_train_sampled1,columns =noclass_train1_cols )
        class_train_sampled1= pd.DataFrame(data=class_train_sampled1,columns =['Class'] )
        data_train2=pd.concat([noclass_train_sampled1, class_train_sampled1], axis=1)
        data_train2.loc[data_train2['xAttack']==5, 'Class'] = 0
        data_train2.loc[data_train2['xAttack']==4, 'Class'] = 1
        noclass_train2 = data_train2.loc[:, data_train2.columns != 'Class']
        class_train2 = data_train2.loc[:, data_train2.columns == 'Class']
        num_class_train2=collections.Counter(class_train2['Class'])
        numofmaj2=num_class_train2[0]
        numofmin2=num_class_train2[1]
        for i2 in B:
            rat2=numofmin2/(numofmaj2*i2)
            print("---------------")
            print("ratio2",i2)
            results['ratio'][a1+1]=i2
            results['Class'][a1]='normal'
            results['Class'][a1+1]='dos'
            results['Class'][a1+2]='probe'
            results['Class'][a1+3]='r2l'
            results['Class'][a1+4]='u2r'
            
            results['Training Datasize'][a1]=traindatasize['normal']
            results['Training Datasize'][a1+1]=traindatasize['dos']
            results['Training Datasize'][a1+2]=traindatasize['probe']
            results['Training Datasize'][a1+3]=traindatasize['r2l']
            results['Training Datasize'][a1+4]=traindatasize['u2r']
            
            results['Testing Datasize'][a1]=testdatasize['normal']
            results['Testing Datasize'][a1+1]=testdatasize['dos']
            results['Testing Datasize'][a1+2]=testdatasize['probe']
            results['Testing Datasize'][a1+3]=testdatasize['r2l']
            results['Testing Datasize'][a1+4]=traindatasize['u2r']
            
            if rat2 <=1:
                ros2 = RandomOverSampler(sampling_strategy=rat2,random_state=5)
                noclass_train_sampled2,class_train_sampled2  = ros2.fit_sample(noclass_train2, class_train2.values.ravel())
                noclass_train_sampled2= pd.DataFrame(data=noclass_train_sampled2,columns =noclass_train1_cols )
                class_train_sampled2= pd.DataFrame(data=class_train_sampled2,columns =['Class'] )
                data_train3=pd.concat([noclass_train_sampled2, class_train_sampled2], axis=1)
                mapping3 = {1 :'normal', 2 :'dos',3: 'probe',4: 'r2l',5: 'u2r'}
                data_train3=data_train3.replace({'xAttack': mapping3})
                data_train3= data_train3.drop(['Class'],axis=1)
                X_train = data_train3.loc[:, data_train3.columns != 'xAttack']
                y_train = data_train3.loc[:, data_train3.columns == 'xAttack']
                
                sampleddatasize=collections.Counter(y_train['xAttack'])
                results['After Sampling'][a1]=sampleddatasize['normal']
                results['After Sampling'][a1+1]=sampleddatasize['dos']
                results['After Sampling'][a1+2]=sampleddatasize['probe']
                results['After Sampling'][a1+3]=sampleddatasize['r2l']
                results['After Sampling'][a1+4]=sampleddatasize['u2r']
                
                clf = AdaBoostClassifier(n_estimators=100, random_state=0,algorithm='SAMME.R')
                #clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0,oob_score=True)
                clf.fit(X_train,y_train.values.ravel())
                y_pred=clf.predict(X_test)
                y_test_arr=np.array(y_test['xAttack'])
                #oobscore=clf.oob_score_
                #print("oob score",oobscore)
                #results['oobscore'][a1]=round(oobscore,5)
                #feature_imp=grid_best_clf.feature_importances_
                #print("feature importances",feature_imp)
                y_act_count=collections.Counter(y_test['xAttack'])
                y_pred_count=collections.Counter(y_pred)
                print("predicted",y_pred_count)
                print("actual",y_act_count)
                results['Predicted Datasize'][a1]=y_pred_count['normal']
                results['Predicted Datasize'][a1+1]=y_pred_count['dos']
                results['Predicted Datasize'][a1+2]=y_pred_count['probe']
                results['Predicted Datasize'][a1+3]=y_pred_count['r2l']
                results['Predicted Datasize'][a1+4]=y_pred_count['u2r']
                y_test_arr=np.array(y_test['xAttack'])
                cn_mat=confusion_matrix( y_pred,y_test['xAttack'])
                print(cn_mat)
                
                TPred_nor=cn_mat[1][1]
                TPred_dos=cn_mat[0][0]
                TPred_probe=cn_mat[2][2]
                TPred_r2l=cn_mat[3][3]
                TPred_u2r=cn_mat[4][4]
                pre_nor=TPred_nor/y_pred_count['normal']
                recall_nor=TPred_nor/y_act_count['normal']
                
                pre_dos=TPred_dos/y_pred_count['dos']
                recall_dos=TPred_dos/y_act_count['dos']
                
                pre_probe=TPred_probe/y_pred_count['probe']
                recall_probe=TPred_probe/y_act_count['probe']
                
                pre_r2l=TPred_r2l/y_pred_count['r2l']
                recall_r2l=TPred_r2l/y_act_count['r2l']
                
                pre_u2r=TPred_u2r/y_pred_count['u2r']
                recall_u2r=TPred_u2r/y_act_count['u2r']
                
                results['Precision'][a1]=round(pre_nor,5)
                results['Recall'][a1]=round(recall_nor,5)
                
                results['Precision'][a1+1]=round(pre_dos,5)
                results['Recall'][a1+1]=round(recall_dos,5)
                
                results['Precision'][a1+2]=round(pre_probe,5)
                results['Recall'][a1+2]=round(recall_probe,5)
                
                results['Precision'][a1+3]=round(pre_r2l,5)
                results['Recall'][a1+3]=round(recall_r2l,5)
                
                results['Precision'][a1+4]=round(pre_u2r,5)
                results['Recall'][a1+4]=round(recall_u2r,5)
                
                F1_nor=2*(pre_nor*recall_nor)/(pre_nor+recall_nor)
                F1_dos=2*(pre_dos*recall_dos)/(pre_dos+recall_dos)
                
                F1_probe=2*(pre_probe*recall_probe)/(pre_probe+recall_probe)
                F1_r2l=2*(pre_r2l*recall_r2l)/(pre_r2l+recall_r2l)
                
                F1_u2r=2*(pre_u2r*recall_u2r)/(pre_u2r+recall_u2r)
                
                
                results['f1-score'][a1]=round(F1_nor,5)
                results['f1-score'][a1+1]=round(F1_dos,5)
                
                results['f1-score'][a1+2]=round(F1_probe,5)
                results['f1-score'][a1+3]=round(F1_r2l,5)
                results['f1-score'][a1+4]=round(F1_u2r,5)
                
                a1=a1+5
                print("a",a1)
                print(classification_report(y_test_arr, y_pred))

        
        
        
        
