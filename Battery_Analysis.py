import streamlit as st
#to add figure
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import scipy as stats
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
#TO DO PCA
from sklearn.decomposition import PCA
from sklearn.pipeline import  make_pipeline
#to do LM
from sklearn.model_selection import train_test_split
#to make linear regession
from sklearn.linear_model import LinearRegression
#to calculate r2, RMSE
from sklearn.metrics import r2_score,mean_squared_error
#TRY interactive plot st
import mpld3
import streamlit.components.v1 as components
#ITERTOOLS
import itertools as itertools
#shuffle select the data in a random mode is "False" because these are time series

my_image=Image.open('Battery.jpg')
my_image=my_image.resize((400,400))
st.image(my_image)

#define functions
def normal(mean, std, color="black"):
    x=np.linspace(mean-6*std, mean+6*std, 200)
    p=stats.norm.pdf(x, mean, std)
    plt.plot(x,p,color,linewidth=2)

option=st.sidebar.selectbox("What do you want to do?",
                            ('Data Pre-exploration',
                             'Principal Components Analysis',
                             'Linear Regression',
                             'Best Subset Selection',
                             'Principal Components Regression'))

df_battery=pd.read_csv('Battery_RUL.csv').dropna()

#add ID number
df_battery['Battery ID']= 0 
batteries=[] 
ID=1
for rul in df_battery['RUL']: 
    batteries.append(ID) 
    if rul == 0: 
        ID+=1
        continue
df_battery['Battery ID'] = batteries

#bar to select battery
selected_id=st.sidebar.selectbox("Battery",df_battery['Battery ID'].unique())

#filter the dataset by ID battery
filtered_df = df_battery[df_battery['Battery ID'] == selected_id]
#show the selected dataset
st.write("""# Selected data""")
st.dataframe(filtered_df)
#%%
if option == 'Data Pre-exploration':
    #mean, sd, Q1,Q2,Q3
    st.write("""Descriptive statistics of the Selected data""")
    filtered_df.iloc[:, 1:-1].describe(include='all').T
    
    num_rows =len(filtered_df)
    st.write(f"Number of cycles: {num_rows}")
    #plot the variableS vs RUL
    st.write("""Diagramms of each variable against RUL """)
    #list of item
    plot_items = list(df_battery.columns)[1:-2]
    #list of ID battery
    batteries = list(df_battery['Battery ID'].unique())
    #plot parameters
    plt.style.use('ggplot') 
    plt.rcParams['figure.figsize']=8,25 
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.rcParams['lines.linewidth'] = 0.5
    fig,ax = plt.subplots(len(plot_items),sharex=True)
    ax[0].invert_xaxis()
    
    

    #for cycle plot battery
    for battery in batteries:
        for i,item in enumerate(plot_items):
            f = sns.lineplot(data=filtered_df[filtered_df['Battery ID']==battery],x='RUL',y=item,
                        color='black',ax=ax[i],
                        )
    st.pyplot(plt)

    #hist RUL
    hist_ret=plt.figure(figsize=(8,6))
    plt.title('RUL, Remaining Useful Time Histogram')
    sns.histplot(filtered_df.RUL,stat='density',kde=True,color='green')
    plt.gca().invert_xaxis()
    st.pyplot(hist_ret)


#Covariance matrix
    cov_mat=filtered_df.iloc[:, 1:-1].cov()
    st.write("""# Covariance matrix """)
    cov_mat
#Correlation matrix
    st.write("""# Correlation matrix """)
    fig=plt.figure(figsize=(10,8))
    sns.heatmap(filtered_df.iloc[:, 1:-1].corr(), annot=True, cmap='Reds',
                center=1, linewidths=.5)
    st.pyplot(fig)

#PCA
#%%
if option=='Principal Components Analysis':
    uns_pca=make_pipeline(StandardScaler(with_std=True),
                  PCA())
    data_pca=filtered_df.iloc[:, 1:-1].copy()
    stds=uns_pca.named_steps['standardscaler']
    uns_pca.fit(data_pca)
    pca=uns_pca.named_steps['pca']
    
    st.write("## PCA results")
    # DataFrame with PC labels and corresponding variance explained
    st.write(""" PCs variance explained ratio""")
    pc_df = pd.DataFrame({'PC': np.arange(1, len(pca.explained_variance_ratio_) + 1),
                      'Variance Explained ratio': pca.explained_variance_ratio_})
    st.dataframe(pc_df,hide_index=True)

    #plot PVE
    fig=plt.figure(figsize=(10,8))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(),
             marker='o',linestyle='--')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.title('Cumulative Explained Variance Ratio by Principal Component')
    st.pyplot(fig)

    #biplot
    biplot_items = list(filtered_df.columns)[1:-1]
    uns_pca2=make_pipeline(StandardScaler(with_std=True),
                  PCA(n_components=2))
    stds2=uns_pca.named_steps['standardscaler']
    uns_pca2.fit(data_pca)
    pca2=uns_pca2.named_steps['pca']
    st.write("""Score Vectors of the first two PCs""")
    PC_scores = pd.DataFrame(pca2.fit_transform(data_pca),
               columns = ['PC 1', 'PC 2'])
    st.dataframe(PC_scores)
    st.write("""Loading Vectors of the first two PCs""")
    PC_Loads = pd.DataFrame(pca2.components_.T, columns=['PC1', 'PC2'], 
                        index=biplot_items)
    st.dataframe(PC_Loads)
    PC1 = PC_scores.iloc[:,0]
    PC2 = PC_scores.iloc[:,1]
    loads = pca2.components_
    scalePC1 = 1.0/(PC1.max() - PC1.min())
    scalePC2 = 1.0/(PC2.max() - PC2.min())
    features = filtered_df.index
    data_pca['Cycle Index']=filtered_df.iloc[:, 0].values
    fig=plt.figure(figsize=(9,6))
    for i, feature in enumerate(biplot_items):
        plt.arrow(0, 0, loads[0, i], 
                loads[1, i],
                color="red",
                head_width=0.03, 
                head_length=0.03)
        plt.text(loads[0, i] * 0.8, 
                loads[1, i] * 0.8, 
                feature, 
                color="red",
                fontsize=12)
 
        sns.scatterplot(x=PC1 * scalePC1,
                    y=PC2 * scalePC2)
 
    plt.xlabel('PC1', fontsize=20)
    plt.ylabel('PC2', fontsize=20)
    plt.title('Biplot', fontsize=20)
    
    for i, label in enumerate(PC_scores.index):
        plt.text(PC1[i] * scalePC1, 
            PC2[i] * scalePC2, str(label), 
            fontsize=10)
    #st.pyplot(fig)
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=650)

#LR
#%%
if option=='Linear Regression':
    data_lm=filtered_df.iloc[:, 1:-2].copy()
    st.write("""Selected data to performe Linear Regression""")
    st.write(data_lm)
    Response = filtered_df['RUL']
    X_train, X_test, y_train, y_test=train_test_split(data_lm,Response, test_size=0.5,
                                                  shuffle=False)
    #Linear regression
    lr=make_pipeline(StandardScaler(with_std=True),LinearRegression())
    print(lr)
    lr_stds=lr.named_steps['standardscaler']
    lr_lr=lr.named_steps['linearregression']
    #fit the data
    lr.fit(X_train,y_train)
    #out of sample predictions
    y_lr_pred=lr.predict(X_test)
    #lr model parameters
    st.write("## LR model parameters")
    st.write(f"Intercept {lr_lr.intercept_:.3f}")
    coefficients_dataset = pd.DataFrame({'Regressor': data_lm.columns, 'Coefficient': lr_lr.coef_})
    st.write(coefficients_dataset)

    st.write("""# In sample and out of sample""")
    st.write("## In sample")
    st.write("In sample dataset")
    df_train_lr = pd.concat([X_train,y_train], axis=1)
    st.dataframe(df_train_lr)
    #squared residual in sample
    resid_lr=(y_train-lr.predict(X_train))**2
    fig=plt.figure(figsize=(7,5))
    plt.scatter(y_train.index,resid_lr,label="LR")
    plt.ylabel("squared residuals")
    plt.title("in sample")
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    #squared residual in sample
    y_lr_fit=lr.predict(X_train)
    Mrss_lr=np.sum(np.square(y_lr_fit - y_train))/X_train.shape[0]
    st.write(f"LR MRSS in sample {Mrss_lr:.3f}")
    st.write(f"LR RMRSS in sample {Mrss_lr**0.5:.3f}")
    st.write(f"LR r-squared in sample {lr.score(X_train,y_train):.3f}")
    st.write("## Out of sample")
    st.write("Out of sample dataset")
    df_test_lr = pd.concat([X_test,y_test], axis=1)
    st.dataframe(df_test_lr)
    #squared error
    error_lr=(y_test-y_lr_pred)**2
    fig=plt.figure(figsize=(7,5))
    plt.scatter(y_test.index,error_lr,label="LR")
    plt.ylabel("squared error")
    plt.title("out of sample")
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    st.write(f"LR MSE out of sample {mean_squared_error(y_test,y_lr_pred):.3f}")
    st.write(f"LR RMSE out of sample {mean_squared_error(y_test,y_lr_pred)**0.5:.3f}")
    st.write(f"LR r-squared out of sample {r2_score(y_lr_pred,y_test):.3f}")

#BSS
#%%
if option=='Best Subset Selection':
    data_lm_bss=filtered_df.iloc[:, 1:-2].copy()
    st.write("""Selected data to performe Linear Regression""")
    data_lm_bss
    Response_bss = filtered_df['RUL']
    X_train_BSS, X_TV_BSS, y_train_BSS, y_TV_BSS=train_test_split(data_lm_bss,Response_bss, test_size=0.5,
                                                  shuffle=False)
    X_Validation_BSS, X_Test_BSS, y_Validation_BSS, y_Test_BSS=train_test_split(X_TV_BSS,y_TV_BSS,
                                                                                 test_size=0.5,
                                                  shuffle=False)
    
    def fit_subset_model(features):
        model = make_pipeline(StandardScaler(with_std=True),LinearRegression())
        model_stds=model.named_steps['standardscaler']
        model_lr=model.named_steps['linearregression']
        model.fit(X_train_BSS.values[:, features], y_train_BSS)
        return model

    def evaluate_subset_model(model, features):
        predictions = model.predict(X_Validation_BSS.values[:, features])
        mse = mean_squared_error(y_Validation_BSS, predictions)
        return mse

    #all possible subsets of features
    n_features = X_train_BSS.shape[1]
    all_subsets = []
    for k in range(1, n_features + 1):
        subsets = itertools.combinations(range(n_features), k)
        all_subsets.extend(subsets)

    best_model = None
    best_subset = None
    best_mse = float('inf')

    # "for" cycle over all subsets and models evalutation
    for subset in all_subsets:
        model = fit_subset_model(subset)
        mse = evaluate_subset_model(model, subset)
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_subset = subset
    
    #out of sample predictions
    y_bssmodel_Vpred=best_model.predict(X_Validation_BSS.values[:,best_subset])

    #features of the best model and corresponding coefficients
    best_features = [f"Feature_{f}" for f in best_subset]
    coefficients = best_model.named_steps['linearregression'].coef_
    feature_names = data_lm_bss.columns

    #names of features in the best subset
    best_feature_names = [feature_names[i] for i in best_subset]

    st.write("## LR BSS model parameters")
    st.write(f"Intercept {best_model.named_steps['linearregression'].intercept_:.3f}")
    df_coefficients = pd.DataFrame({'Feature': best_feature_names, 'Coefficient': coefficients})
    st.write(df_coefficients)
    

    st.write("""# Train set, Validation set and Test set""")
    st.write("## Train")
    st.write("Train dataset")
    df_train_BSS = pd.concat([X_train_BSS,y_train_BSS], axis=1)
    st.dataframe(df_train_BSS)
    #squared residual in sample
    y_bssmodel_Tpred=best_model.predict(X_train_BSS.values[:,best_subset])
    Mrss=np.sum(np.square(y_bssmodel_Tpred - y_train_BSS))/X_train_BSS.shape[0]
    
    resid_lr_BSS=(y_train_BSS-best_model.predict(X_train_BSS.values[:,best_subset]))
    fig=plt.figure(figsize=(7,5))
    plt.scatter(y_train_BSS.index,resid_lr_BSS,label="LR_BSS")
    plt.ylabel("squared residuals")
    plt.title("Train set")
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    st.write(f"LR MRSS BSS model, Train set {Mrss:.3f}")
    st.write(f"LR RMRSS BSS model, Train set {Mrss**0.5:.3f}")
    st.write(f"LR r-squared BSS model, Train set {best_model.score(X_train_BSS.values[:,best_subset],y_train_BSS):.3f}")

    #Validation set
    st.write("## Validation")
    st.write("Validation dataset")
    df_V_BSS = pd.concat([X_Validation_BSS,y_Validation_BSS], axis=1)
    st.dataframe(df_V_BSS)    
    error_lr_BSS_V=(y_Validation_BSS-y_bssmodel_Vpred)
    fig=plt.figure(figsize=(7,5))
    plt.scatter(y_Validation_BSS.index,error_lr_BSS_V**2,label="LR_BSS")
    plt.ylabel("squared error in validation set")
    plt.title("Validation set")
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    st.write(f"LR BSS MSE model, Validation set {mean_squared_error(y_Validation_BSS,y_bssmodel_Vpred):.3f}")
    st.write(f"LR BSS RMSE model, Validation set {mean_squared_error(y_Validation_BSS,y_bssmodel_Vpred)**0.5:.3f}")
    st.write(f"LR r-squared model, BSS Validation set {best_model.score(X_Validation_BSS.values[:,best_subset],y_Validation_BSS):.3f}")
 
    #Test set
    st.write("## Test")
    st.write("Test datset")
    df_T_BSS = pd.concat([X_Test_BSS,y_Test_BSS], axis=1)
    st.dataframe(df_T_BSS)  
    y_bssmodel_Tpred=best_model.predict(X_Test_BSS.values[:,best_subset])

    error_lr_BSS_T=(y_Test_BSS-y_bssmodel_Tpred)
    fig=plt.figure(figsize=(7,5))
    plt.scatter(y_Test_BSS.index,error_lr_BSS_T**2,label="LR_BSS")
    plt.ylabel("squared error")
    plt.title("Test set")
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    st.write(f"LR BSS MSE model, Test set {mean_squared_error(y_Test_BSS,y_bssmodel_Tpred):.3f}")
    st.write(f"LR BSS RMSE model, Test set {mean_squared_error(y_Test_BSS,y_bssmodel_Tpred)**0.5:.3f}")
    st.write(f"LR r-squared BSS model, Test set {r2_score(y_bssmodel_Tpred,y_Test_BSS):.3f}")

#PCR
#%%
if option=='Principal Components Regression':
        
    data_pcr=filtered_df.iloc[:, 1:-2].copy()
    st.write("""Selected data to performe Principal Components Regressor""")
    st.write(data_pcr)
    Response_pcr = filtered_df['RUL']
    X_train_pcr, X_test_pcr, y_train_pcr, y_test_pcr=train_test_split(data_pcr,Response_pcr, test_size=0.5,
                                                  shuffle=False)
    n_comp=3
    st.write(f"Number of PCA components {n_comp}")
    #to put thogheter different methods, make_pipeline
    pcr=make_pipeline(StandardScaler(with_std=True),
                  PCA(n_components=n_comp),
                  LinearRegression())

    pcr.fit(X_train_pcr,y_train_pcr)
    stds=pcr.named_steps['standardscaler']
    pca_r=pcr.named_steps['pca']
    lr_pcr=pcr.named_steps['linearregression']
    st.write("# PCR results")
    # DataFrame with PC labels and corresponding variance explained
    st.write(""" PCs variance explained ratio""")
    pcr_df = pd.DataFrame({'PC': np.arange(1, len(pca_r.explained_variance_ratio_) + 1),
                      'Variance Explained ratio': pca_r.explained_variance_ratio_})
    st.dataframe(pcr_df,hide_index=True)

    y_pcr_pred=pcr.predict(X_test_pcr)

    st.write("## PCR model parameters")
    st.write(f"Intercept {lr_pcr.intercept_:.3f}")

    pcr_coef = pd.DataFrame({'PC':np.arange(1, pca_r.n_components_ + 1), 'Coefficient': lr_pcr.coef_})
    st.dataframe(pcr_coef,hide_index=True)
    
    y_pcr_Trainpred=pcr.predict(X_train_pcr)
    Mrss_pcr=np.sum(np.square(y_pcr_Trainpred - y_train_pcr))/X_train_pcr.shape[0]
    
    st.write("""# In sample and out of sample""")
    st.write("## In sample")
    st.write("In sample dataset")
    df_train_pcr = pd.concat([X_train_pcr,y_train_pcr], axis=1)
    st.dataframe(df_train_pcr)
    #squared residual in sample
    resid_pcr=(y_train_pcr-y_pcr_Trainpred)**2
    fig=plt.figure(figsize=(7,5))
    plt.scatter(y_train_pcr.index,resid_pcr,label="PCR")
    plt.ylabel("squared residuals")
    plt.title("in sample")
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    
    st.write(f"PCR MRSS BSS Train set {Mrss_pcr:.3f}")
    st.write(f"PCR RMRSS BSS train set {Mrss_pcr**0.5:.3f}")
    st.write(f"PCR BSS r-squared Train set {pcr.score(X_train_pcr,y_train_pcr):.3f}")
    
    st.write("## Out of sample")
    st.write("Out of sample dataset")
    df_test_pcr = pd.concat([X_test_pcr,y_test_pcr], axis=1)
    st.dataframe(df_train_pcr)
    #squared residual out of sample
    error_pcr=(y_test_pcr-y_pcr_pred)**2
    fig=plt.figure(figsize=(7,5))
    plt.scatter(y_test_pcr.index,error_pcr,label="PCR")
    plt.ylabel("squared error")
    plt.title("out of sample")
    plt.legend()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    st.write(f"PCR MSE out of sample {mean_squared_error(y_test_pcr,y_pcr_pred):.3f}")
    st.write(f"PCR RMSE out of sample {mean_squared_error(y_test_pcr,y_pcr_pred)**0.5:.3f}")
    st.write(f"PCR r-squared out of sample {r2_score(y_test_pcr,y_pcr_pred):.3f}")