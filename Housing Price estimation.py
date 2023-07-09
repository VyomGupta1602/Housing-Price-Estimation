import copy, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[167]:


dataset=pd.read_csv("HousingPrice.csv")
# In[197]:


from sklearn.preprocessing import LabelEncoder
for x in dataset:
    dataset["mainroad"] = LabelEncoder().fit_transform(dataset["mainroad"])
    dataset["guestroom"] = LabelEncoder().fit_transform(dataset["guestroom"])
    dataset["basement"] = LabelEncoder().fit_transform(dataset["basement"])
    dataset["furnishingstatus"] = LabelEncoder().fit_transform(dataset["furnishingstatus"])

# In[200]:


x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

# In[203]:


dataset.isna().sum()


# In[204]:


dataset.hist(bins=10, figsize=(10, 10))


# In[205]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[211]:


b_init = 909.909840909
w_init = np.array([ 0.39133572,0.39133535, 18.39133535, 0.10003020, 0.42131618, 0.39133535, 0.10003020, 0.42131618])


# In[214]:


def compute_cost(x, y, w, b): 
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)    
    return cost


# In[215]:


cost = compute_cost(x_train, y_train, w_init, b_init)
print(cost)


# In[219]:


def dwdb(x, y, w, b): 
    m = x.shape[0]
    dj_dw = np.zeros((n,))
    c=0.0
    for i in range(m):                                
        f_wb_i = np.dot(x[i], w)+b
        for j in range(n):
            dj_dw[j] = dj_dw[j]+(f_wb_i-y[i])*x[i,j]
        c+=(f_wb_i - y[i])
    dj_dw =  dj_dw/ (m)
    c=c/(m)
    return dj_dw,c


# In[220]:


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = w_in.copy()
    b = b_in
    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(X, y, w, b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db      
        if i<100000:
            J_history.append( cost_function(X, y, w, b))
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history


# In[234]:


initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-8
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,n= x_test.shape
ypred2=[]
for i in range(m):
    ypred2.append(np.dot(x_test[i], w_final)+b_final)


# In[235]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()

# In[206]:


from sklearn.linear_model import LinearRegression
Regression = LinearRegression()
Regression.fit(x_train, y_train)


# In[207]:


y_pred = Regression.predict(x_test)





