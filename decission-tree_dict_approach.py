# Use decisiontree.pretty_print() to visualize the decision tree
# May display warning messages but output is displayed
# toLabel() and label() functions in the class are given as skeleton code in the hint video of the project

from sklearn import datasets

import numpy as np
import math
import pprint
import pandas as pd
from operator import itemgetter

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

    # Function to print details of each node in the tree levelwise
    def level_print(self):
        self.level_nodes = sorted(self.level_nodes, key=itemgetter('level'))

        dup_levels = self.level_nodes.copy()

        while dup_levels != []:
            node = dup_levels.pop(0)
            self.dict_print(node)


    #Function to buid tree which is implemented using dictionary
    def buildTree(self,df,out,cols,l=0,tree={}):
        # Checking there are no columns to split upon,if yes then insert this node as leaf node into dictionary 'tree'
        if len(cols) == 0:
            tree['leaf'] = True
            tree['feature'] = "Reached leaf node"
            tree['level'] = l
            tree['entropy'] = 0.0
            df['out'] = out.copy()
            tree['count'] = df['out'].value_counts().to_dict()
            uni = df['out'].unique()

            if len(uni)==1:
                tree['out'] = uni[0]
            else:
                tree['out'] = list(uni)

            self.level_nodes.append(tree.copy())

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

            self.level_nodes.append(tree.copy())

            return

        # if tree is an empty dictionary then calculate the details of the maximum gain node from the entered data frame
        if tree == {}:
            # Assigning tree with the details of the feature with maximum gain
            tree[node] = {'gain':df['gain/'+node].unique()[0],'feature':node,'entropy':df['ent/'+node].unique()[0],'level':l,'count':df['out'].value_counts().to_dict(),'leaf':False}
            tree['leaf'] = False

            self.level_nodes.append(tree[node].copy())

            cols = set(cols).difference(set([node]))
            l += 1
            #for each unique class in the feature calling build_tree function recursively to get each node to form the tree
            for f in df[node].unique():
                tree[node][f] = {}
                subtable = self.get_subtable(df, node, f)
                columns = subtable.columns.values

                self.buildTree(subtable[columns[:4]].copy(), subtable[columns[4]].copy(), list(cols), l=l, tree=tree[node][f])

        #return the final tree
        return tree

    #Function to filter and train data for building the tree
    def fit(self,df,out):
        #Calling the filter function
        df = self.filter(df)
        #Conversion to make sure any numeric column index of the dataframe are converted to string
        columns = [str(f) for f in df.columns.values]
        #Storing the converted column indicies to class variable
        self.columns = columns
        #repalcing the column indicies of the dataframe with the converted column indicies
        df.columns = columns

        # List to strore the each node of the tree while building a tree
        self.level_nodes = []

        self.root = self.buildTree(df, out, columns)

    #Function to visualize the decision tree dictionary
    def pretty_print(self):
        pprint.pprint(self.root)

    #Function to label the numeric data into some unique values
    def label(self,val,feature):
        #selecting the throttle values of the specified feature to label the numeric data as unique values
        boundaries = []
        for i in range(len(self.boundaries)):
            b = self.boundaries[i]
            key = list(b.keys())[0]
            if key == feature:
                boundaries = b[feature]
                break
        
        #labelling the data as unique value based on the selected throttle values
        if (val < boundaries[0]):
            return 'a'
        elif (val < boundaries[1]):
            return 'b'
        elif (val < boundaries[2]):
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
        
        #store the throttle value of each feature in the form of list of dictionaries
        d = {old_feature_name:[first,second,third]}
        self.boundaries.append(d)

        return df[old_feature_name].apply(lambda x: self.label(x,old_feature_name))

    #Function to find and convert columns with numeric datatypes to uniquely labeled values
    def filter(self,df):
        #Finding the datatypes present in each column and storing it as dictionary...
        #where column index is the key and datatype pesent in the column as value and storing it as class variable
        self.column_types = dict(df.dtypes)
        # list to store throttle values of each feature in the form of list of dictionaries
        self.boundaries = []
        for d in self.column_types.keys():
            if self.column_types[d] == float or self.column_types[d] == int:
                df[d] = self.toLabel(df.copy(),d)

        return df

    #Function to find out and convert keys with numeric values to uniquely labeled data for predicting the output
    def filter_target(self,dty):
        for d in dty.keys():
            if type(dty[d]) == int or type(dty[d]) == float:
                dty[d] = self.label(dty[d],d)

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

        # If all the features entered by the user is present in the then start traversing through the tree to predict the final output...
        # else display the error message
        if present:

            # Calling filter_target function to convert any numeric class in the dictionary to labelled data to get prediction
            dty = self.filter_target(dty)

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
    dtree.fit(data.copy(),target)
    #Printing the nodes of the tree levelwise by calling level print function
    dtree.level_print()
    #input values for predicting the output
    d = {'sepal_length':1.5,'sepal_width':2.2,'petal_length':3.2,'petal_width':1.3}
    # Input for predicting the output
    print()
    print("Input for predicting output :",d)
    #Predicted output for the given input dictionary
    pred = dtree.predict(d)
    #Printing the predicted output
    print()
    print("Predicted Output:",pred)

#Calling the test function
test()