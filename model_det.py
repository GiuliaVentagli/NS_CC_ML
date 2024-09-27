import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW

##########################
def renormalize(x, k2min, k2max):
    return (x + abs(k2min))/(k2max + abs(k2min))

##########################
def produce_model_class(n_nodes, nodes1, nodes2, nodes3, activation_hidden, ke_init, activation_out,
                              ke_init_out, n_features):
    model = Sequential()
    model.add(Dense(nodes1, activation = activation_hidden, kernel_initializer = ke_init,
                    input_shape = (n_features,)))
    if n_nodes>=2:
        model.add(Dense(nodes2, activation = activation_hidden, kernel_initializer = ke_init))
        if n_nodes>=3:
            model.add(Dense(nodes3, activation = activation_hidden, kernel_initializer = ke_init))
            
    model.add(Dense(2, activation = activation_out, kernel_initializer = ke_init_out))
    
    return model

##########################
def produce_model_reg(nodes, activation_hidden, ke_init, activation_out, 
                              ke_init_out, n_features):
    model = Sequential()
    model.add(Dense(nodes, activation = activation_hidden, kernel_initializer = ke_init, 
                    input_shape = (n_features,)))
    model.add(Dense(15, activation = activation_out, kernel_initializer = ke_init_out))
    return model

##########################
def store_values_class(loss_list, val_loss_list, acc_list, val_acc_list, history):
    
    loss_list.append(history.history['loss'][-1])
    val_loss_list.append(history.history['val_loss'][-1])
    acc_list.append(history.history['binary_accuracy'][-1])
    val_acc_list.append(history.history['val_binary_accuracy'][-1])

##########################    
def store_values_reg(loss_list, val_loss_list, history):
    
    loss_list.append(history.history['loss'][-1])
    val_loss_list.append(history.history['val_loss'][-1])

##########################
def plot_learning_curve_class(history, rep, points, nodes1, nodes2, nodes3, check_k2 = True):
        pyplot.figure()
        pyplot.title('Learning Curves')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()
        if check_k2 == True:
            pyplot.savefig(f'LearningCurvesClassk2_n{rep}_N{points}_{nodes1}_{nodes2}_{nodes3}.png')
        else:
            pyplot.savefig(f'LearningCurvesClass_n{rep}_N{points}_{nodes1}_{nodes2}_{nodes3}.png')
            
##########################
def plot_learning_curve_reg(history, rep, points, nodes, check_k2 = True):
        pyplot.figure()
        pyplot.title('Learning Curves')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()
        if check_k2 == True:
            pyplot.savefig(f'LearningCurvesRegk2_n{rep}_N{points}_{nodes}.png')
        else:
            pyplot.savefig(f'LearningCurvesReg_n{rep}_N{points}_{nodes}.png')

##########################
def plot_accuracy(history, rep, points, nodes1, nodes2, nodes3, check_k2 = True):
        pyplot.figure()
        pyplot.title('Accuracy')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Binary_accuracy')
        pyplot.plot(history.history['binary_accuracy'], label='train')
        pyplot.plot(history.history['val_binary_accuracy'], label='val')
        pyplot.legend()
        pyplot.show()
        if check_k2 == True:
            pyplot.savefig(f'AccuracyClassk2_n{rep}_N{points}_{nodes1}_{nodes2}_{nodes3}.png')
        else:
            pyplot.savefig(f'AccuracyClass_n{rep}_N{points}_{nodes1}.png')

##########################            
def plot_mean_cs(stat_test, stat_hat, points, save = False):
    pyplot.scatter(stat_test[:,0],stat_hat[:,0], s=5)
    pyplot.xlim(0,1)
    pyplot.ylim(0,1)
    pyplot.xlabel('c_s real') 
    pyplot.ylabel('c_s pred')
    pyplot.title('N=%s' % points)
    if save == True:
        pyplot.savefig(f'c_smeanN%s.png' %points)
        
##########################
def produce_mean_cs_separate(data_test, data_hat, stat_test, stat_hat, rep_test):
    i_in, i_fin = 0, rep_test
    
    for i in range(0,int(len(stat_test)/rep_test)):
        step_test = DescrStatsW(stat_test[i_in:i_fin,0], weights=(1/stat_test[i_in:i_fin,1]**2))
        data_test[i]=[step_test.mean,step_test.std]
        
        step_hat = DescrStatsW(stat_hat[i_in:i_fin,0], weights=(1/stat_hat[i_in:i_fin,1]**2))
        data_hat[i]=[step_hat.mean,step_hat.std]
        
        i_in += rep_test
        i_fin += rep_test

##########################
def plot_mean_cs_separate(data_test, data_hat, points, save = False):
    pyplot.scatter(data_test[:,0],data_hat[:,0], s=1)
    pyplot.errorbar(data_test[:,0],data_hat[:,0], yerr = data_hat[:,1], fmt ='.',
                    markersize='2', ecolor='orange')
    pyplot.xlim(0,1)
    pyplot.ylim(0,1)
    pyplot.xlabel('<c_s> real') 
    pyplot.ylabel('<c_s> pred')
    pyplot.title('N=%s' %points)
    if save == True:
        pyplot.savefig(f'c_smeanN%serror.png' %points)

##########################
def produce_mean_cs_overall(indices, data_join, data_test, data_hat):
    i_in = 0
    for i in range(0,len(indices)):
        step_test = DescrStatsW(data_join[i_in:indices[i]+1,0],
                                weights=(1/data_join[i_in:indices[i]+1,1]**2))
        data_test[i]=[step_test.mean,step_test.std]
        
        step_hat = DescrStatsW(data_join[i_in:indices[i]+1,2], 
                               weights=(1/data_join[i_in:indices[i]+1,3]**2))
        data_hat[i]=[step_hat.mean,step_hat.std]
        
        i_in = indices[i]+1

##########################
def plot_mean_cs_overall(data_test, data_hat, points, save = False):
    line = [0.39,0.8]
    #pyplot.scatter(data_test[:,0],data_hat[:,0], s=5)
    pyplot.errorbar(data_test[:,0],data_hat[:,0], yerr = 3.0*data_hat[:,1], fmt ='.',
                    markersize='4', color='black', ecolor='black',elinewidth=1)
    pyplot.plot(line,line, color='grey',linewidth=1,linestyle='dashed')
    pyplot.xlim(0.39,0.8)
    pyplot.ylim(0.39,0.8)
    pyplot.xlabel('<c_s> real') 
    pyplot.ylabel('<c_s> pred')
    pyplot.title('N=%s' %points)
    if save == True:
        pyplot.savefig(f'c_smeanmeanN%serror.png' %points)

##########################
def plot_cs_profile(j, y_test, y_hat, rep_test, points, save = False):
    pyplot.plot(y_test[j,1:8],y_test[j,8:15],color='black',linewidth=1.5, label="real")
    for i in range(j,j+rep_test):
        pyplot.plot(y_hat[i,1:8],y_hat[i,8:15],linewidth=1)
        pyplot.xlim(0,1)
        pyplot.ylim(0,1)
        pyplot.xlabel('ρ/ρ_f',fontsize=16) 
        pyplot.ylabel('c_s',fontsize=16)
        pyplot.title(f'c_s profile, N={points}, n={rep_test}')
        pyplot.legend()
    if save == True:
        pyplot.savefig(f'c_sprofileN{points}n{rep_test}.png')

##########################        
def save_EOS(j, y_test, yeos_test, y_hat, yeos_hat, rep_test, points, save = True):
    list_to_save = y_test[j,:] #save into list EOS parameters (Λ,ρ,cs)
    list_to_save = np.insert(list_to_save, 0, yeos_test[j], axis=0) #add AP4/SLy class
    for i in range(0,rep_test):
        list_hat = y_hat[j+i,:]
        list_hat = np.insert(list_hat, 0, yeos_hat[j+i], axis=0)
        
        list_to_save = np.vstack((list_to_save, list_hat))
    if save == True:
        np.savetxt(f'EOSj{j}N{points}n{rep_test}.txt', list_to_save)
    #return(list_to_save)