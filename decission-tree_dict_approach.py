# Use decisiontree.pretty_print() to visualize the decision tree
# May display warning messages but output is displayed
# toLabel() and label() functions in the class are given as skeleton code in the hint video of the project

from sklearn import datasets

import numpy as np
import math
import pprint
import pandas as pd

#Class for implementing decision trees
class decision_tree :
    #Function to find out the entropy of given dataframe and its column index
    def entropy(self,col,name):
        tot = col.shape[0]
        uni = col[name].value_counts().rename_axis('unique').reset_index(name='counts')
        uni['p'] = uni['counts']/tot
        uni['p'] = uni['p'].apply(lambda x: -1*(x*math.log2(x)))

        return uni['p'].sum()
    #Function to find out the column with maximum gain ratio out of the given dataframe and it's respective column index
    def gain(self,x,y,features):
        #appending output column with y dataframe as it's new feature
        x['out'] = y
        #calculating entropy of the output column in the dataframe by calling the entropy function
        x['ent/out'] = self.entropy(x,'out')
        #finding the unique values and it's count in the output of the dataframe
        target_values = x['out'].value_counts().to_dict()
        # finding the initial size of the samples before splitting
        initial_size = x.shape[0]

        #iterating through each feature in the dataframe and calculating entropy,information gain and gain ratio of each feature and appending this value as a new...
        # column to the dataframe
        for f in features:
            values = x[f].value_counts().to_dict()
            ent_x = 0
            split_info = 0
            for k1 in values.keys():
                current_size = x[x[f] == k1].shape[0]
                ent1 = 0

                for k2 in target_values.keys():
                    df1 = x[(x[f] == k1)&(x['out']==k2)].shape[0]
                    if df1 != 0:
                        ent1 += (-df1/current_size)*math.log2(df1/current_size)

                ent_x += (current_size/initial_size)*ent1
                split_info += (-current_size/initial_size)*math.log2(current_size/initial_size)
            x['ent/'+f] = ent_x
            x["inf/"+f] = x['ent/out']-ent_x
            x['gain/'+f] = x['inf/'+f]/split_info

        return x,self.find_winner(x)

    #Function to find the column with maximum gain ratio where given the gain ratio of all columns
    def find_winner(self,df):
        features = df.columns.values
        df_details = df[features[5:]]
        df_details_cols = df_details.columns.values

        best_f = {}
        #forming a dictionary with feature as key and gain as value from the dataframe
        for f in df_details_cols:
            if f.find("gain/") != -1:
                best_f[f.replace("gain/","")] = df[f].unique()[0]
        #sorting the dictionary keys with maximum value i.e., gain
        sorted_dict = list(dict(sorted(best_f.items(),key=lambda item: item[1])).keys())
        #return the key i.e., feature with maximum gain
        return sorted_dict[0]

    #Function to get the subtable from a dataframe given the dataframe,column index and a unique value in the column
    def get_subtable(self,df, node, value):
        return df[df[node] == value]

    #Function to print out the details of each node in the decision tree
    def dict_print(self,dty):
        print()
        level = dty.get('level')
        print('level :',level)
        count = dty.get('count')
        for k in count.keys():
            print('count of',k,":",count.get(k))
        print("Entropy :",round(dty.get('entropy'),2))
        print('feature :',dty.get('feature'))
        gain = dty.get('gain')
        if gain != None:
            print('gain :',round(dty.get('gain'),2))

    #Function to buid tree which is implemented using dictionary
    def buildTree(self,df,out,cols,l=0,printer=False,tree={}):
        # Checking there are no columns to split upon,if yes then insert this node as leaf node into dictionary 'tree'
        if len(cols) == 0:
            tree['leaf'] = True
            tree['feature'] = "Reached leaf node"
            tree['level'] = l
            tree['entropy'] = 0.0
            # dfc = df.copy()
            df['out'] = out.copy()
            tree['count'] = df['out'].value_counts().to_dict()
            uni = df['out'].unique()

            if len(uni)==1:
                tree['out'] = uni[0]
            else:
                tree['out'] = list(uni)

            if printer:
                self.dict_print(tree)

            return

        #Find the feature and the gain of the feature with maximum gain from the dataframe by calling the gain function
        df,node = self.gain(df.copy(),out.copy(),cols)

        #checking if a node is pure node, if yes insert this node as root node in the dictionary 'tree'
        if len(df['out'].unique()) == 1:
            tree['leaf'] = True
            tree['feature'] = "Reached leaf node"
            tree['level'] = l
            tree['entropy'] = 0.0
            tree['out'] = df['out'].unique()[0]
            tree['count'] = df['out'].value_counts().to_dict()

            if printer:
                self.dict_print(tree)

            return

        # if tree is an empty dictionary then calculate the details of the maximum gain node from the entered data frame
        if tree == {}:
            # Assigning tree with the details of the feature with maximum gain
            tree[node] = {'gain':df['gain/'+node].unique()[0],'feature':node,'entropy':df['ent/'+node].unique()[0],'level':l,'count':df['out'].value_counts().to_dict(),'leaf':False}
            tree['leaf'] = False
            if printer:
                self.dict_print(tree[node])

            cols = set(cols).difference(set([node]))
            l += 1
            #for each unique class in the feature calling build_tree function recursively to get each node to form the tree
            for f in df[node].unique():
                tree[node][f] = {}
                subtable = self.get_subtable(df, node, f)
                columns = subtable.columns.values

                if printer:
                    self.buildTree(subtable[columns[:4]], subtable[columns[4]], list(cols), l=l, tree=tree[node][f],printer=True)
                else:
                    self.buildTree(subtable[columns[:4]].copy(),subtable[columns[4]].copy(),list(cols),l=l,tree=tree[node][f])

        #return the final tree
        return tree

    #Function to filter and train data for building the tree
    def fit(self,df,out,printer=False):
        #Calling the filter function
        df = self.filter(df)
        #Conversion to make sure any numeric column index of the dataframe are converted to string
        columns = [str(f) for f in df.columns.values]
        #Storing the converted column indicies to class variable
        self.columns = columns
        #repalcing the column indicies of the dataframe with the converted column indicies
        df.columns = columns

        if printer:
            self.root = self.buildTree(df, out, columns,printer=True)
        else:
            self.root = self.buildTree(df, out, columns)

        print()

        return self.root

    #Function to visualize the decision tree dictionary
    def pretty_print(self):
        pprint.pprint(self.root)

    #Function to label the numeric data into some unique values
    def label(self,val):
        #Uses throttle value in the class variable 'boundaries' to label the entered numeric data
        if (val < self.boundaries[0]):
            return 'a'
        elif (val < self.boundaries[1]):
            return 'b'
        elif (val < self.boundaries[2]):
            return 'c'
        else:
            return 'd'

    #Function to find the boundaries based on which the pandas dataframe columns will be labeled with unique values
    def toLabel(self,df, old_feature_name):
        second = df[old_feature_name].mean()
        minimum = df[old_feature_name].min()
        first = (minimum + second) / 2
        maximum = df[old_feature_name].max()
        third = (maximum + second) / 2

        #setting the throttle values to label the numeric data as list of elements in the class variable 'booundaries'
        self.boundaries = []
        self.boundaries.append(first)
        self.boundaries.append(second)
        self.boundaries.append(third)

        return df[old_feature_name].apply(lambda x: self.label(x))

    #Function to find and convert columns with numeric datatypes to uniquely labeled values
    def filter(self,df):
        #Finding the datatypes present in each column and storing it as dictionary...
        #where column index is the key and datatype pesent in the column as value and storing it as class variable
        self.column_types = dict(df.dtypes)

        for d in self.column_types.keys():
            if self.column_types[d] == float or self.column_types[d] == int:
                df[d] = self.toLabel(df.copy(),d)

        return df

    #Function to find out and convert keys with numeric values to uniquely labeled data for predicting the output
    def filter_target(self,dty):
        for d in dty.keys():
            if type(dty[d]) == int or type(dty[d]) == float:
                dty[d] = self.label(dty[d])

        return dty

    #Function to valildate input values and predict the output
    def predict(self,dty):
        # setting the copy of the root node to local variable to work further to get the output prediction
        nroot = self.root.copy()

        # Converting the keys of the dictionary to string to improve compliance with tree
        cols = [str(i) for i in dty.keys()]
        # Check if all the input features are present in the tree
        present = True
        feature = ""
        for c in cols:
            if c not in self.columns:
                feature = c
                present = False
                break

        # Calling filter_target function to convert any numeric class in the dictionary to labelled data to get prediction
        dty = self.filter_target(dty)

        # If all the features entered by the user is present in the then start traversing through the tree to predict the final output...
        # else display the error message
        if present:
            while nroot.get('leaf') == False:
                key = set(list(nroot.keys())).intersection(set(cols))
                key = list(key)[0]
                # print(list(key)[0])
                nroot = nroot.get(key)
                value = dty.get(key)

                if nroot.get(value) != None:
                    nroot = nroot.get(value)
                else:
                    print("the value",value,"is not present in the feature",key ,"for the given combination")
                    return
        else:
            print("the",feature,"is not present in the decission tree")

        # Return the output after traversing the tree till the leaf node
        return nroot.get('out')







#Function for testing our decision_tree class
def test():
    #loading the iris_data
    iris_data = datasets.load_iris()
    #loading the target names of the iris_data
    target_names = list(iris_data.target_names)
    #column labels for the iris data set
    cols = ["sepal_length", "sepal_width", 'petal_length', 'petal_width']
    #converting the iris data set to pandas dataframe
    data = pd.DataFrame(iris_data.data, columns=cols)
    # loading and converting iris_data to pandas dataframe
    target = pd.DataFrame(iris_data.target)[0].apply(lambda x: target_names[x]).to_frame()

    # Initializing decision tree
    dtree = decision_tree()
    #Training the decision tree for the iris data
    tree = dtree.fit(data.copy(),target,printer=True)
    #input values for predicting the output
    d = {'sepal_length':1.5,'sepal_width':1.2,'petal_length':1.2,'petal_width':1.3}
    # Input for predicting the output
    print("Input for predicting output :",d)
    #Predicted output for the given input dictionary
    pred = dtree.predict(d)
    #Printing the predicted output
    print("Predicted Output:",pred)

#Calling the test function
test()