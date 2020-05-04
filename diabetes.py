#x_data loader

import numpy as np
np.seterr(all='raise')
import code
class Diabetes_Binary_Classifier:
    def __init__(self,provided_method='lr'):#default method is logistic regression with no regularization
        self.x_data=[]
        self.y_data=[]
        self.w=[]
        self.training_dataset_index=[]
        self.test_dataset_index=[]        
        self.method=provided_method
    def _parse_line(self,line_string):
        _Ls=line_string.split('  ')#_Ls[0]=\pm 1,Ls[1]=feature_vector
        assert(len(_Ls)==2)
        float_feature_vector=[float(float_string.split(':')[1]) for float_string in _Ls[1].split(' ')]
        #code.interact(local=locals())                  
        #append float_feature_vector to numpy x_data array
        self.x_data.append(float_feature_vector)
        self.y_data.append(int(_Ls[0])/2+0.5)
    def _logistic_function(self,x):
        try:
            lf=1/(1+np.exp(-np.dot(self.w,x)))
        except FloatingPointError:
            code.interact(local=locals())
        return lf
        #self.w and x are of narray type
    def load(self,file_name):
        for line_string in open(file_name).read().split('\n'):
            self._parse_line(line_string)
        self.feature_dim=len(self.x_data[0])
        #output statistics to console
        print("parsed: %s,x_data count:%d,feature_dim:%d"%(file_name,len(self.x_data),self.feature_dim))
        self.x_data=np.array(self.x_data)
        #normalization of each column,[0,1]
        for i in range(self.feature_dim):
            min_tmp=min(self.x_data[:,i])
            max_tmp=max(self.x_data[:,i])
            self.x_data[:,i]=(self.x_data[:,i]-min_tmp)/(max_tmp-min_tmp)
    def training_test_split(self,percertage=0.8):
        #split self.(x,y)data into training and test dataset
        #select 80% len(self.x_data) as training data
        self.training_dataset_index=set(np.random.choice(len(self.x_data),int(percertage*len(self.x_data)),replace=False))
        self.test_dataset_index=[i for i in range(len(self.x_data)) if i not in self.training_dataset_index]
        #the remaining data is left as test dataset
        
    def cross_validation_setup(self):
        #split the total data into five amounts
        self.five_cv=[[],[],[],[],[]]
        choice_left=list(range(len(self.x_data)))
        for i in range(5):
            selected_num=int(0.2*len(self.x_data))
            if(i==4):
                selected_num=len(choice_left)
            self.five_cv[i]=set(np.random.choice(choice_left,selected_num,replace=False))
            choice_left=list(set(choice_left).difference(self.five_cv[i]))
            
        
    def logistic_regression(self,turning_parameter=1):
        #initialize self.w,with dimension equal to len(self.x_data[0])
        self.w=np.zeros(self.feature_dim)
        #process training data one by one, multiple pass
        
        #we should use measurement of error rate in training_dataset to determine when to stop
        
        #many parameters are empirical, such as how many turns the training process should take and
        #how to select the turning_parameter, alpha
        iterative_count=0
        last_empirical_error_count=0
        empirical_error_count=len(self.x_data[:,0]) 
        while(True):
            last_empirical_error_count=empirical_error_count
            empirical_error_count=0            
            for i in self.training_dataset_index:
                if(np.dot(self.w,self.x_data[i,:])*(self.y_data[i]*2-1)<0):
                    empirical_error_count+=1
                self.w+=turning_parameter*(self.y_data[i]-self._logistic_function(self.x_data[i,:]))*self.x_data[i,:]
            if(last_empirical_error_count<=empirical_error_count):
                break
            self.w/=np.linalg.norm(self.w)
            iterative_count+=1
            #maybe there is no need for the normalization of self.w
            #self.w=self.w/np.linalg.norm(self.w)
            #report the error rate at this pass
        #code.interact(local=locals())
        #print("Logistic Regression, use %d times to converge"%iterative_count)        
        return
    def IRLS_cross_validate(self,regularization_parameter):                
        #five turn
        error_rate=[]
        for i in range(5):
            #in each turn, assemble training and test set firstly
            self.training_dataset_index=[]
            for j in range(5):
                if(i==j):
                    #assemble test set
                    self.test_dataset_index=self.five_cv[i]
                else:      
                    self.training_dataset_index.extend(self.five_cv[j])
            #then run IRLS for the given regularization_parameter
            self.IRLS(regularization_parameter)
            error_rate.append(self._predict()/len(self.test_dataset_index))
        #print('IRLS cross validate for regularization parameter: %f,\n Average error rate for test set: %f'%(regularization_parameter,np.mean(error_rate)))
        return np.mean(error_rate)
        
    def IRLS(self,regularization_parameter=0):
        #Iterative Reweighted Least Square uses Newton-Raphson iterative method to solve nonlinear function
        self.w=np.zeros(self.feature_dim)        
        update_vector=np.zeros(self.feature_dim)
        #calculate gradient and Hessian matrix
        iterative_count=0
        last_empirical_error_count=0
        empirical_error_count=len(self.x_data[:,0])         
        while(True):
            last_empirical_error_count=empirical_error_count
            empirical_error_count=0                    
            gradient=update_vector*regularization_parameter
            Hessian=regularization_parameter*np.identity(self.feature_dim)        
            for i in self.training_dataset_index:
                _logistic_temp=self._logistic_function(self.x_data[i,:]);
                gradient+=(self.y_data[i]-_logistic_temp)*self.x_data[i,:]
                Hessian+=_logistic_temp*(_logistic_temp-1)*np.kron(self.x_data[i,:],self.x_data[i,:]).reshape((self.feature_dim,self.feature_dim))
                if(np.dot(self.w,self.x_data[i,:])*(self.y_data[i]*2-1)<=0):
                    empirical_error_count+=1                
            update_vector=-np.linalg.solve(Hessian,gradient)
            #if update_vector is very small, stop updating process
            if(last_empirical_error_count<=empirical_error_count):
                break
            self.w+=update_vector 
            iterative_count+=1
        #report the iteration times
        #print("IRLS, use %d times to converge"%iterative_count)
        #print("last_empirical_error_count: %f"%last_empirical_error_count)
        #print("empirical_error_count: %f"%empirical_error_count)        
        return last_empirical_error_count   
        
    def SVM(self,isLinear=True,gamma=1):        
        import svmutil
        #libsvm wrapper
        #first reload the data in svm format
        y_svm=[int(self.y_data[i]*2-1) for i in self.training_dataset_index];
        x_svm=[]
        for i in self.training_dataset_index:
            x_svm_item={}
            for j in range(self.feature_dim):
                x_svm_item[j]=self.x_data[i,j]
            x_svm.append(x_svm_item)
        #generate test data
        y_svm_test=[int(self.y_data[i]*2-1) for i in self.test_dataset_index];
        x_svm_test=[]
        for i in self.test_dataset_index:
            x_svm_test_item={}
            for j in range(self.feature_dim):
                x_svm_test_item[j]=self.x_data[i,j]
            x_svm_test.append(x_svm_test_item)        
        error_rate=[]
        c_discrete_log=[(-5+2*i) for i in range(10)]
        #fake for testing
        #return [c_discrete_log,c_discrete_log]
        for i in c_discrete_log:
            if(isLinear):
                svm_train_str='-t 0 -c %f'%(np.exp(i))
            else:
                svm_train_str='-c %f -g %f'%(np.exp(i),gamma)
            libsvm_model = svmutil.svm_train(y_svm,x_svm,svm_train_str)
            _, p_acc, _ = svmutil.svm_predict(y_svm_test, x_svm_test, libsvm_model)
            error_rate.append((100-p_acc[0])/100)
        #error report: (100-p_acc[0])/100
        return [c_discrete_log,error_rate]      
    def text_report(self):
        #generate tabular report
        return        
    def graphic_report(self):
        #generate graphic report with matplotlib
        import matplotlib.pyplot as plt
        #(task=='logistic_regression_regularization_parameter_tuning'):
        if(False):
            rp,er=self.logistic_regression_regularization_parameter_tuning()
            plt.plot(rp,er,'ro',rp,er)
            plt.xlabel('regularization_parameter')
            plt.ylabel('error rate')
            plt.title('regularization_parameter_tuning')
            plt.savefig('logistic_regression_regularization_parameter_tuning.eps')        
            plt.show()
        #(task=='typical_curve_of_loss_and_accurary'):
        if(False):
            pt,er1,er2=self.typical_curve_of_loss_and_accurary()
            plt.plot(pt,er1,'ro',pt,er2,'b^')
            line_1,=plt.plot(pt,er1,'r-',label='train')
            line_2,=plt.plot(pt,er2,'b-',label='test')
            plt.xlabel('percertage of traing_set')
            plt.ylabel('error rate')
            plt.legend(handles=[line_1,line_2])            
            plt.title('typical_curve_of_loss_and_accurary')            
            plt.savefig('typical_curve_of_loss_and_accurary.eps')        
            plt.show()
        #(task=='SVM hyper-parameter tuning')
        if(True):
            #get linear svm plot data
            self.training_test_split()            
            plt.rc('text', usetex=True)
            x_log,er=self.SVM()
            er_Gaussian=[]
            gamma_discrete=[np.power(2.0,1.0*(-7+2*i)) for i in range(6)]
            Guassian_color=['b', 'g', 'c', 'm', 'y', 'k']
            line_bundle=[]
            for i in gamma_discrete:
                _,er_temp=self.SVM(False,i)
                er_Gaussian.append(er_temp)
            plt.plot(x_log,er,'ro')
            linear_line,=plt.plot(x_log,er,'r-',label='linear svm')
            line_bundle.append(linear_line)
            for index,i in enumerate(gamma_discrete):
                plt.plot(x_log,er_Gaussian[index],Guassian_color[index]+'o')
                tmp_line,=plt.plot(x_log,er_Gaussian[index],Guassian_color[index]+'-',label='RBF,$\gamma=%E$'%(gamma_discrete[index]))
                line_bundle.append(tmp_line)
            plt.legend(handles=line_bundle)      
            plt.xlabel('regularization\_parameter C(log scale)')
            plt.ylabel('error rate')
            plt.title('SVM\_regularization\_parameter\_tuning')
            plt.savefig('SVM_regularization_parameter_tuning.eps')        
            plt.show()
            
            code.interact(local=locals())        
        return
    def typical_curve_of_loss_and_accurary(self):
        pt=[0.4,0.5,0.6,0.7,0.8,0.9]
        er_1=[]
        er_2=[]
        self.training_dataset_index=set(np.random.choice(len(self.x_data),int(pt[0]*len(self.x_data)),replace=False))
        self.test_dataset_index=[i for i in range(len(self.x_data)) if i not in self.training_dataset_index]        
        for d in pt:
            er_1.append(self.IRLS()/len(self.training_dataset_index))
            er_2.append(self._predict()/len(self.test_dataset_index))
            #add 10% to the training dataset index
            self.training_dataset_index=set(np.random.choice(self.test_dataset_index,int(0.1*len(self.x_data)),replace=False))
            self.test_dataset_index=[i for i in range(len(self.x_data)) if i not in self.training_dataset_index]
            # we can not random split training and test set in each iteration!
            #self.training_test_split(d)            
        return [pt,er_1,er_2]
    def logistic_regression_regularization_parameter_tuning(self):
        print("\n********* Logistic Regression, regularization_parameter tuning process:*********\n")    
        self.cross_validation_setup()
        rp=[np.log(i+1) for i in range(10)]
        er=[]
        for i in rp:
            er.append(self.IRLS_cross_validate(i));
        return [rp,er]
        
    def _predict(self):#use the trained "w" to predict on test dataset
        empirical_error_count=0
        for i in self.test_dataset_index:
            if(np.dot(self.w,self.x_data[i,:])*(self.y_data[i]*2-1)<0):
                empirical_error_count+=1
        return empirical_error_count
        
    def predict(self):
        #first report the behaviour of algorithm on training dataset
        empirical_error_count=0
        for i in self.training_dataset_index:
            if(np.dot(self.w,self.x_data[i,:])*(self.y_data[i]*2-1)<0):
                empirical_error_count+=1            
        print("On Training data,error rate: %f"%(empirical_error_count/len(self.test_dataset_index)))        
        empirical_error_count=self._predict()
        #report the statitical result to stdout
        print("\n predict_result:\n \ttest_data_count: %s,\
            \n\t empirical_error count:%d,\
            \n\t empirical_error rate:%f"%(len(self.test_dataset_index),empirical_error_count,empirical_error_count/len(self.test_dataset_index)))
    def _error_report(self,method,turns=10):
        error_vector=[]
        print("\n********* Method: %s:*********\n"%method)
        for i in range(turns):
            self.training_test_split()        
            if(method=='logistic_regression'):
                self.logistic_regression()
            elif(method=='IRLS'):
                self.IRLS()
            error_vector.append(self._predict())
        error_rate_mean=np.mean(error_vector)/len(self.test_dataset_index)
        error_rate_var=np.var(error_vector)/(len(self.test_dataset_index)**2)
        print("\t error_rate_mean:%f,error_rate_std_var:%f"%(error_rate_mean,np.sqrt(error_rate_var)))
# use the following code to debug
#              
if __name__ == "__main__":
    diabetes_binary_classifier_instance=Diabetes_Binary_Classifier()
    diabetes_binary_classifier_instance.load('diabetes.txt')
    iteration_time=100    
    print("\n********* Method: Logistic Regression:*********\n")
    er=[]
    for i in range(iteration_time):
        diabetes_binary_classifier_instance.training_test_split()
        diabetes_binary_classifier_instance.logistic_regression()
        er.append(diabetes_binary_classifier_instance._predict()/len(diabetes_binary_classifier_instance.test_dataset_index))
    print("err mean= %f, err std var= %f"%(np.mean(er),np.sqrt(np.var(er))))    

    print("\n********* Method: Iterative Reweighted Least Square:*********\n")    
    er=[]
    for i in range(iteration_time):
        diabetes_binary_classifier_instance.training_test_split()
        diabetes_binary_classifier_instance.IRLS()
        er.append(diabetes_binary_classifier_instance._predict()/len(diabetes_binary_classifier_instance.test_dataset_index))
    print("err mean= %f, err std var= %f"%(np.mean(er),np.sqrt(np.var(er))))    

    
    diabetes_binary_classifier_instance.graphic_report()
    
    
