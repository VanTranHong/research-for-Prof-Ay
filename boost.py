def boost (classifier, n_est, rate):
    abc = AdaBoostClassifier(n_estimators = n_est, base_estimator=classifier, learning_rate =rate)
    return abc

def boost_svm(data, column, features, method, params, kernel):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {}
    n_ests = [25,50,75,100]
    rates = np.linspace(0.1,0.5,5)
    for n_est in n_ests:
        for rate in rates:
            name = str(n_est)+','+str(rate)
            dict.update({name:[0,0,0,0]})
    
    
    
    if kernel =='poly' or 'linear'
        parameters = params.split(',')
        c= float(parameters[0])
        gamma = float(parameters[1])
        
        for train, test in skf.split(X,y):
            train_data, test_data = X[train], X[test]  
            train_data, test_data = impute(train_data, test_data)
            train_result, test_result = y[train], y[test] 
            train_data = pd.DataFrame(data = train_data, columns=columns)
            test_data = pd.DataFrame(data = test_data, columns=columns)
            features_selected = run_feature_selection(method,train_data,train_result,features, 'svm')
        
            for feature in features_selected:
                top_features[feature]+=1 
            train_data = np.array(train_data[features_selected])
            
            test_data = np.array(test_data[features_selected])
            for n_est in n_ests:
                for rate in rates:
                    name = str(n_est)+','+str(rate) 
                    estimator = SVC(probability=True,kernel = kernel, C=c, gamma=gamma)
                    boost = boost(estimator, n_est,rate)  
                    boost.fit(train_data,train_result)
                    predicted_label = boost.predict(test_data)
                    tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                    convert_matrix = [tn,fp,fn,tp]
                    get_matrix = dict.get(name)
                    result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                    dict.update({name: result})
        
    
    else:
        parameters = params.split(',')
        c= float(parameters[0])
        gamma = float(parameters[1])
        degree = int(parameters[2])
        for train, test in skf.split(X,y):
            train_data, test_data = X[train], X[test]  
            train_data, test_data = impute(train_data, test_data)
            train_result, test_result = y[train], y[test] 
            train_data = pd.DataFrame(data = train_data, columns=columns)
            test_data = pd.DataFrame(data = test_data, columns=columns)
            features_selected = run_feature_selection(method,train_data,train_result,features, 'svm')
        
            for feature in features_selected:
                top_features[feature]+=1 
            train_data = np.array(train_data[features_selected])
            
            test_data = np.array(test_data[features_selected])
            for n_est in n_ests:
                for rate in rates:
                    name = str(n_est)+','+str(rate) 
                    estimator = SVC(probability=True,kernel = kernel, C=c, gamma=gamma, degree = degree)
                    boost = boost(estimator, n_est,rate)  
                    boost.fit(train_data,train_result)
                    predicted_label = estimator.predict(test_data)
                    tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                    convert_matrix = [tn,fp,fn,tp]
                    get_matrix = dict.get(name)
                    result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                    dict.update({name: result})
    return dict
        
        
    
    
    
    
def boost_rdforest(data, column, features, method, params):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {}
    n_ests = [25,50,75,100]
    rates = np.linspace(0.1,0.5,5)
    for n_est in n_ests:
        for rate in rates:
            name = str(n_est)+','+str(rate)
            dict.update({name:[0,0,0,0]})
    parameters = params.split(',')
    estimator= int(parameters[0])
    max_feature = int(parameters[1])
        
    for train, test in skf.split(X,y):
        train_data, test_data = X[train], X[test]  
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test] 
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'svm')
        
        for feature in features_selected:
            top_features[feature]+=1 
        train_data = np.array(train_data[features_selected])
            
        test_data = np.array(test_data[features_selected])
        for n_est in n_ests:
            for rate in rates:
                name = str(n_est)+','+str(rate) 
                estimator = RandomForestClassifier(n_estimators=estimator, max_features=max_feature)
                boost = boost(estimator, n_est,rate)  
                boost.fit(train_data,train_result)
                predicted_label = estimator.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
    return dict
    
def boost_naive_bayes(data, column, features, method, params):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict_Gauss = {}
    dict_Bernoulli = {}
    n_ests = [25,50,75,100]
    rates = np.linspace(0.1,0.5,5)
    
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
    for train, test in skf.split(X,y):
        
        train_data, test_data = X[train], X[test]
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'naive_bayes')
       
        
        for feature in features_selected:
            top_features[feature]+=1
            
        
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        for n_est in n_ests:
            for rate in rates:
                name = str(n_est)+','+str(rate) 
                estimator = GaussianNB()
                boost = boost(estimator, n_est,rate)  
                boost.fit(train_data,train_result)
                predicted_label = estimator.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict_Gauss.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict_Gauss.update({name: result})
                
                estimator = BernoulliNB()
                boost = boost(estimator, n_est,rate)  
                boost.fit(train_data,train_result)
                predicted_label = estimator.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict_Bernoulli.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict_Bernoulli.update({name: result})
                
def boost_knn(data, column, features, method, params):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict={}
    n_ests = [25,50,75,100]
    rates = np.linspace(0.1,0.5,5)
    neighbor = int(params)
    
    
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
    for train, test in skf.split(X,y):
        
        train_data, test_data = X[train], X[test]
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'naive_bayes')
       
        
        for feature in features_selected:
            top_features[feature]+=1
            
        
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        for n_est in n_ests:
            for rate in rates:
                name = str(n_est)+','+str(rate) 
                estimator = KNeighborsClassifier(n_neighbors=neighbor)
                boost = boost(estimator, n_est,rate)  
                boost.fit(train_data,train_result)
                predicted_label = estimator.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
                
                
                

    
    
