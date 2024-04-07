import numpy as np
import pandas as pd
import cv2  as cv
import os

class Data():
    def __init__(self,data_home):
        # data_home: the local path storing the downloaded data
        self.data_home=data_home

    def load_minist(self,n_class=4):
        #download url: https://www.openml.org/d/40996, Fashion-MNIST
        #n_class: the number of classes to be used
        from sklearn.datasets import fetch_openml
        X, y  = fetch_openml('Fashion-MNIST', return_X_y=True, data_home='./data/')
        X = X.to_numpy()
        y = [int(item) for item in list(y)]
        y = np.array(y)
        if n_class < 10:
            idx = y < n_class
            X, y = X[idx], y[idx]
        return X, y

    def load_USPS(self):
        #download url: https://www.openml.org/d/41964
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml('USPS', return_X_y=True, data_home='./data/')
        X = X.to_numpy()
        y = np.array(y)
        for i in range(len(list(y))):
            y[i] = int(y[i].strip())
        return X, y

    def load_autoUniv_au6_1000(self):
        #download url: https://www.openml.org/d/1555, size: 1000 *41
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml('autoUniv-au6-1000', return_X_y=True, data_home='./data/')
        X=X.to_numpy()
        y=np.array(y)
        V12_dict=self.str2int(X[:,11])
        V31_dict = self.str2int(X[:,30])
        V40_dict = self.str2int(X[:,39])
        y_dict=self.str2int(y)
        for x in X:
            x[11]=V12_dict[x[11].strip()]
            x[30] = V31_dict[x[30].strip()]
            x[39] = V40_dict[x[39].strip()]

        for i in range(len(list(y))):
            y[i] = y_dict[y[i].strip()]

        return X, y

    def load_gender(self):
        #download url: https://www.kaggle.com/datasets/muhammadtalharasool/simple-gender-classification
        #size: 132*7
        dir='./data/manifold_learning/gender.csv'
        X=pd.read_csv(dir)
        X=X.to_numpy()
        y = X[:, 0]
        X=X[:,1:9]

        f1=self.str2int(X[:, 3])
        f2 =self.str2int(X[:, 4])
        f3 =self.str2int(X[:, 5])
        f4 = self.str2int(X[:, 7])
        y_dict = self.str2int(y)
        for x in X:
            x[3]=f1[x[3].strip()]
            x[4] = f2[x[4].strip()]
            x[5] = f3[x[5].strip()]
            x[7] = f4[x[7].strip()]
        for i in range(len(list(y))):
            y[i] = y_dict[y[i].strip()]
        return X,y

    def load_student_evaluation(self):
        #download url: https://www.kaggle.com/datasets/csafrit2/higher-education-students-performance-evaluation
        #size: 146*31
        dir = './data/manifold_learning/student_prediction.csv'
        X = pd.read_csv(dir)
        X = X.to_numpy()
        y = X[:, 32]
        X = X[:, 1:32]
        return X,y

    def load_User_Knowledge(self):
        #download url: https://www.kaggle.com/datasets/fafiliam/user-knowledge?resource=download&select=data_student.csv
        #size: 259*5
        dir = './data/manifold_learning/User_Knowledge.csv'
        X = pd.read_csv(dir)
        X = X.to_numpy()
        y = X[:, 5]
        X = X[:, 0:5]
        y_dict = self.str2int(y)
        for i in range(len(list(y))):
            y[i] = y_dict[y[i].strip()]
        return X, y

    def load_yeast(self):
        #download url: https://www.kaggle.com/datasets/samanemami/yeastcsv
        #size: 1484*9
        dir = './data/manifold_learning/yeast.csv'
        X = pd.read_csv(dir)
        X = X.to_numpy()
        y = X[:, 8]
        X = X[:, 0:8]
        y_dict = self.str2int(y)
        for i in range(len(list(y))):
            y[i] = y_dict[y[i].strip()]
        return X, y

    def str2int(self,x):
        #convert string to int. some datasets have string labels need to be converted to int labels
        x_set = set(list(x))
        X_set_redup=set([])
        for i in x_set:
            X_set_redup.add(i.strip())

        x_dict = {}
        cnt = 0
        for f in X_set_redup:
            x_dict[f] = cnt
            cnt += 1
        return x_dict

    def load_coil20(self):
        dir = './data/latent_graph_learning/coil-20/'
        X = []
        y = []
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                img = cv.imread(os.path.join(dirpath, filename), cv.IMREAD_GRAYSCALE)
                img_resized = cv.resize(img, (32, 32))
                X.append(img_resized.flatten().tolist())
                y.append(int(dirpath.split('/')[-1]))
        X = np.array(X)
        y = np.array(y)
        return X, y

    def load_orl(self):
        dir = './data/latent_graph_learning/ORL/'
        nps=[]
        for i in range(40):
            np_=[]
            for dirpath, dirnames, filenames in os.walk(dir+'s'+str(i+1)):
                for filename in filenames:
                    try:
                        img = cv.imread(os.path.join(dirpath, filename), cv.IMREAD_GRAYSCALE)
                        img_resized = cv.resize(img, (32, 32))
                    except:
                        print('img error: ', filename)
                        continue
                    np_.append(img_resized.flatten().tolist())

            nps.append(np_)

        X=np.vstack(tuple(nps))
        ys = []
        for i in range(40):
            ys.append(i*np.ones(len(nps[i])))
        y = np.vstack(tuple(ys)).flatten().astype(int)
        return X, y

    def load_yale(self):
        from PIL import Image
        dir = './data/latent_graph_learning/YALE/'
        X = []
        y = []
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                image = Image.open(os.path.join(dirpath, filename)).convert("L")
                img_resized=Image.Image.resize(image, (32, 32))
                # array = np.array(image)
                # img = cv.imread(os.path.join(dirpath, filename), cv.IMREAD_GRAYSCALE)
                # img_resized = cv.resize(img, (32, 32))
                X.append(np.array(img_resized).flatten().tolist())
                y.append(int(filename[7:9])-1)

        X = np.array(X)
        y = np.array(y)
        return X, y

    def load_BA(self):
        dir = './data/latent_graph_learning/BA.mat'
        import scipy.io as sio
        data = sio.loadmat(dir)
        X=[]
        y=[]
        cnt=0
        for cs in data['dat']:
            for ele in cs:
                X.append(ele.flatten().tolist())
                y.append(cnt)
            cnt+=1
        X = np.array(X)
        y = np.array(y)
        return X, y

    def load_data(self,data_name):
        if data_name=='minist':
            return self.load_minist()
        elif data_name=='USPS':
            return self.load_USPS()
        elif data_name=='autoUniv_au6_1000':
            return self.load_autoUniv_au6_1000()
        elif data_name=='gender':
            return self.load_gender()
        elif data_name=='student_prediction':
            return self.load_student_evaluation()
        elif data_name=='User_Knowledge':
            return self.load_User_Knowledge()
        elif data_name=='yeast':
            return self.load_yeast()
        elif data_name=='coil20':
            return self.load_coil20()
        elif data_name=='orl':
            return self.load_orl()
        elif data_name=='yale':
            return self.load_yale()
        elif data_name=='BA':
            return self.load_BA()
        else:
            print('Dataset %s doesn\'t exist')
