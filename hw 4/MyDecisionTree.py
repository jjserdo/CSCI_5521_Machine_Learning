import numpy as np
from scipy import stats as st

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """

    def __init__(
        self,
    ):
        self.is_leaf = False  # whether or not the current node is a leaf node
        self.feature = None  # index of the selected feature (for non-leaf node)
        self.class_label = None  # class label (for leaf node)
        self.left_child = None  # left child node
        self.right_child = None  # right child node


class Decision_tree:
    """
    Decision tree with binary features
    """

    def __init__(self, min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self, train_x, train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x, train_y, self.min_entropy)

    def predict(self, test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype(
            "int"
        )  # placeholder
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            ''' # mine start '''
            node = self.root
            while node.class_label is None:
                if test_x[i][node.feature] == 0:
                    node = node.left_child
                else:
                    node = node.right_child
            prediction[i]=node.class_label
            ''' # mine end '''
        return prediction

    def generate_tree(self, data, label, min_entropy):
        # initialize the current tree node
        cur_node = Tree_node()
    
        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)
    
        # determine if the current node is a leaf node
        if node_entropy < min_entropy:
            # determine the class label for leaf node
            ''' # Imine start '''
            A, B = st.mode(label)
            cur_node.class_label = A[0]
            ''' # mine end '''
            return cur_node
    
        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data, label)
        cur_node.feature = selected_feature
    
        # split the data based on the selected feature and start the next level of recursion
        ''' # mine start '''
        cur_node.left_child = self.generate_tree(data[data[:,selected_feature]==0],label[data[:,selected_feature]==0],min_entropy)
        cur_node.right_child = self.generate_tree(data[data[:,selected_feature]==1],label[data[:,selected_feature]==1],min_entropy)
        ''' # mine end '''
    
        return cur_node
    
    
    def select_feature(self, data, label):
        # iterate through all features and compute their corresponding entropy
        ''' # mine start '''
        best_feat = 0
        be = self.compute_node_entropy(label)
        for i in range(len(data[0])):
            left_y = label[data[:,i]==0]
            right_y = label[data[:,i]==1]
            # compute the entropy of splitting based on the selected features
            cur_entropy = self.compute_split_entropy(
                left_y, right_y
            )  # You need to replace the placeholders ('None') with valid inputs
    
            # select the feature with minimum entropy
            if (cur_entropy < be):
                be = cur_entropy
                best_feat = i
        ''' # mine end '''        
        return best_feat
    
    def compute_split_entropy(self, left_y, right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two splits
        ''' # mine start '''
        l = len(left_y)
        r = len(right_y)
        N = l + r
        split_entropy = l/N * self.compute_node_entropy(left_y) + r/N * self.compute_node_entropy(right_y)
        ''' # mine end '''
        return split_entropy
    
    
    def compute_node_entropy(self, label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        ''' # mine start '''
        N = len(label) # gives out the total number of labels
        A, uniques = list(np.unique(label, return_counts=True)) # gives out the number 
        node_entropy = 0
        for i in range(len(uniques)):
            p = uniques[i]/N
            node_entropy  -= p*np.log2(p+1e-15)
        ''' # mine end '''
        return node_entropy
