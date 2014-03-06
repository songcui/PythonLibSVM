import numpy as np
import random

def svm_gd(file_feature, file_target, learning_rate, C, convergence_criteria):

    ## The algorithm implements gradient decent method for linear SVM optimization. The algorithm reads
    ## the training datapoints from file_feature and target values assigned for each data point from file_target.
    ## C is the parameter for the softmargin penalty in the cost function of the SVM
    
    f = open(file_feature,'r')
    lines = f.readlines()
    number_of_sample = len(lines)
    number_of_dimension = len(lines[0].split(','))
    data_point = np.zeros((number_of_sample, number_of_dimension))
    f.close()
    count = 0
    for line in lines:
        listOfFeatures = line.split(',')
        data_point[count,:] = np.array([float(x) for x in listOfFeatures])
        count+=1

    del lines

    target = np.zeros((number_of_sample))
    f = open(file_target, 'r')
    lines = f.readlines();

    ## The algorithm only proceeds when the number of data points is the same as the number of target values assigned for the data points
    
    if number_of_sample!=len(lines):
        print("Inconsistent sample size!")
    else:
        count = 0
        for line in lines:
            target[count] = float(line)
            count+=1

        shuffled_sequence = np.array(range(number_of_sample))
        random.shuffle(shuffled_sequence)

        target = np.array([target[k] for k in shuffled_sequence])
        data_point = np.array([data_point[k,:] for k in shuffled_sequence])

    ## zero initianization for w and b
        
        w = np.zeros((number_of_dimension))
        b = float(0.0)
        residue = []
        residue0 = np.dot(w,w)
        residue_array = np.array([max(0, 1-target[j]*(np.dot(w, data_point[j,:])+b)) for j in range (number_of_sample)])
        
    ## calculate the cost function
        
        residue.append(1/2*residue0 + C*sum(residue_array))
        count = 0
        error = []
        error.append(1.0)
        while(error[len(error)-1]>convergence_criteria):
            if count==0:           
                error[len(error)-1]=0
            w = w - w*learning_rate;    
            for j in range (number_of_sample):
                w1  = np.array([C*target[j]*data_point[j,k] if (np.dot(data_point[j,:], w)+b)*target[j]<1 else 0 for k in range (number_of_dimension)])
                if (np.dot(data_point[j,:], w)+b)*target[j]<1:
                    b1  = C*target[j]
                else:
                    b1 = 0
                b = b+ b1*learning_rate
                w = w + w1*learning_rate
            residue0 = np.dot(w,w)
            residue_array = np.array([max(0, 1-target[j]*(np.dot(w, data_point[j,:])+b)) for j in range (number_of_sample)])    
            residue.append(1/2*residue0 + C*sum(residue_array))
            new_error = abs(residue[len(residue)-1]-residue[len(residue)-2])/residue[len(residue)-2]*100
            error.append(new_error)
            print(new_error)
            count+=1
            
        residue = np.array(residue)
    return w, b, residue

def svm_sgd(file_feature, file_target, learning_rate, C, convergence_criteria):
    ## The algorithm implements stochastic gradient decent method for linear SVM optimization. The algorithm reads
    ## the training datapoints from file_feature and target values assigned for each data point from file_target.
    ## C is the parameter for the softmargin penalty in the cost function of the SVM
    
    f = open(file_feature,'r')
    lines = f.readlines()
    number_of_sample = len(lines)
    number_of_dimension = len(lines[0].split(','))
    data_point = np.zeros((number_of_sample, number_of_dimension))
    f.close()
    count = 0
    for line in lines:
        listOfFeatures = line.split(',')
        data_point[count,:] = np.array([float(x) for x in listOfFeatures])
        count+=1

    del lines

    target = np.zeros((number_of_sample))
    f = open(file_target, 'r')
    lines = f.readlines();

    ## The algorithm only proceeds when the number of data points is the same as the number of target values assigned for the data points
    
    if number_of_sample!=len(lines):
        print("Inconsistent sample size!")
    else:
        count = 0
        for line in lines:
            target[count] = float(line)
            count+=1

        shuffled_sequence = np.array(range(number_of_sample))
        random.shuffle(shuffled_sequence)

        target = np.array([target[k] for k in shuffled_sequence])
        data_point = np.array([data_point[k,:] for k in shuffled_sequence])

    ## zero initianization for w and b
        
        w = np.zeros((number_of_dimension))
        b = float(0.0)
        residue = []
        residue0 = np.dot(w,w)
        residue_array = np.array([max(0, 1-target[j]*(np.dot(w, data_point[j,:])+b)) for j in range (number_of_sample)])
        
    ## calculate the cost function
        residue.append(1/2*residue0 + C*sum(residue_array))
        count = 0
        error = []
        error.append(1.0)
        while(error[len(error)-1]>convergence_criteria):
            if count==0:           
                error[len(error)-1]=0
            for j in range (number_of_sample):
                w1  = np.array([-w[k]+C*target[j]*data_point[j,k] if (np.dot(data_point[j,:], w)+b)*target[j]<1 else -w[k] for k in range (number_of_dimension)])
                if (np.dot(data_point[j,:], w)+b)*target[j]<1:
                    b1  = C*target[j]
                else:
                    b1 = 0;
                b = b+ b1*learning_rate;
                w = w + w1*learning_rate;
            residue0 = np.dot(w,w)
            residue_array = np.array([max(0, 1-target[j]*(np.dot(w, data_point[j,:])+b)) for j in range (number_of_sample)]);    
            residue.append(1/2*residue0 + C*sum(residue_array))
            new_error = abs(residue[len(residue)-1]-residue[len(residue)-2])/residue[len(residue)-2]*100
            error.append(new_error)
            print(new_error)
            count+=1

        residue = np.array(residue)
    return w, b, residue
