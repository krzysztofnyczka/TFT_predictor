import numpy as np
from sklearn.utils import resample
import graphviz

def entropy(counts):
    counts = counts/sum(counts)
    return -np.sum(counts * np.log2(counts + 1e-100))

def gini(counts):
    counts = counts/sum(counts)
    return 1 - np.sum(counts * counts)

def mean_err_rate(counts):
    counts = counts/sum(counts)
    return 1 - max(counts)

class AbstractSplit:
    """Split the examples in a tree node according to a criterion.
    """
    def __init__(self, attr):
        self.attr = attr

    def __call__(self, x):
        """Return the subtree corresponding to x."""
        raise NotImplementedError

    def build_subtrees(self, df, subtree_kwargs):
        """Recuisively build the subtrees."""
        raise NotImplementedError

    def iter_subtrees(self):
        """Return an iterator over subtrees."""
        raise NotImplementedError

    def add_to_graphviz(self, dot):
        """Add the split to the graphviz vizluzation."""
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}: {self.attr}"

class CategoricalMultivalueSplit(AbstractSplit):
    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = {}
        for group_name, group_df in df.groupby(self.attr):
            child = Tree(group_df, **subtree_kwargs)
            self.subtrees[group_name] = child

    def __call__(self, x):
        # Return the subtree for the given example
        attr = self.attr
        if x[attr] in self.subtrees:
            return self.subtrees[x[attr]]
        return None

    def iter_subtrees(self):
        return self.subtrees.values()
    
    def add_to_graphviz(self, dot, parent, print_info):
        for split_name, child in self.subtrees.items():
            child.add_to_graphviz(dot, print_info)
            dot.edge(f'{id(parent)}', f'{id(child)}',
                     label=f'{split_name}')

def get_categorical_split_and_purity(df, parent_purity, purity_fun, attr,
                                     normalize_by_split_entropy=False):
    """Return a multivariate split and its purity.
    Args:
        df: a dataframe
        parent_purity: purity of the parent node
        purity_fun: function to compute the purity
        attr: attribute over whihc to split the dataframe
        normalize_by_split_entropy: if True, divide the purity gain by the split
            entropy (to compute https://en.wikipedia.org/wiki/Information_gain_ratio)
    
    Returns:
        pair of (split, purity_gain)
    """
    split = CategoricalMultivalueSplit(attr)
    # Compute the purity after the split
    purity = 0
    for group_name, group_df in df.groupby(attr):
        # purity of all values in descending order
        purity += purity_fun(group_df['target'].value_counts()) * len(group_df)
        
    purity /= len(df)
    purity_gain = parent_purity - purity
    if normalize_by_split_entropy:
        purity_gain /= entropy(df[attr].value_counts())
    return split, purity_gain

def get_split(df, criterion='infogain', nattrs=None):
    # Implement termination criteria:
    # 1. Node is pure
    target_value_counts = df['target'].value_counts()
    if len(target_value_counts) == 1:
        return None
    # 2. No split is possible
    #    First get a list of attributes that can be split
    possible_splits = [c for c in df.columns if c != 'target' and len(df[c].value_counts()) > 1]
    # specified nattrs number
    if nattrs is not None:
        np.random.shuffle(possible_splits)
        possible_splits = possible_splits[:nattrs]
    #    Terminate early if none are possivle
    if not possible_splits:
        return None
    
    # Get the base purity measure and the purity function
    if criterion in ['infogain', 'infogain_ratio']:
        purity_fun = entropy
    elif criterion in ['mean_err_rate']:    
        purity_fun = mean_err_rate
    elif criterion in ['gini']:
        purity_fun = gini
    else:
        raise Exception("Unknown criterion: " + criterion)
    base_purity = purity_fun(target_value_counts)

    best_purity_gain = -1
    best_split = None

    # Random Forest support
    # Randomize the split by restricting the number of attributes
    
    for attr in possible_splits:
        if np.issubdtype(df[attr].dtype, np.number):
            # Handling of numerical attributes will be defined later, in a manner 
            # similar to categorical ones
            split_sel_fun = get_numrical_split_and_purity
        else:
            split_sel_fun = get_categorical_split_and_purity
        
        split, purity_gain = split_sel_fun(
            df, base_purity, purity_fun, attr,
            normalize_by_split_entropy=criterion.endswith('ratio'))
        
        if purity_gain > best_purity_gain:
            best_purity_gain = purity_gain
            best_split = split
    return best_split

class Tree:
    def __init__(self, df, **kwargs):
        super().__init__()
        # Assert that threre are no missing values,
        # TODO: remove this for bonus problem #XXX
        assert not df.isnull().values.any()
        
        # We need to let subrees know about all targets to properly color nodes
        if 'all_targets' not in kwargs:
            kwargs['all_targets'] = sorted(df['target'].unique())
        # Save keyword arguments to build subtrees
        kwargs_orig = dict(kwargs)
        
        # Get kwargs we know about, remaning ones are for splitting
        self.all_targets = kwargs.pop('all_targets')
        
        # Save debug info for visualization
        self.counts = df['target'].value_counts()
        self.info = {
            'num_samples': len(df),
            'entropy': entropy(self.counts),
            'gini': gini(self.counts)
        }
        
        self.split = get_split(df, **kwargs)
        if self.split:
            #print('!!S', self.split)
            self.split.build_subtrees(df, kwargs_orig)

    def upper_confidence_interval(self, f, N, z=0.5):
        # http://chrome.ws.dei.polimi.it/images/6/62/IRDM2015-04-DecisionTreesPruning.pdf?fbclid=IwAR2j1xK_WTsF77rUucQW1q-y09s3EWHgfX52H6_3hXO_MTyGVSs5Fsoi1Sc
        return (f + ((z ** 2) / (2 * N)) + z * ((f / N - (f ** 2) / N + z ** 2 / (4 * (N ** 2))) ** 0.5)) / (1 + (z ** 2) / N)
            

    def confidence_interval_pruning(self):
        if self.split:
            for c in self.split.iter_subtrees():
                c.confidence_interval_pruning()

        n = self.info["num_samples"]
        current_error = self.counts / np.sum(self.counts)
        current_error = list(sorted(list(current_error), reverse=True))
        current_error = self.upper_confidence_interval(np.sum(current_error[1:]), n)
        # print(current_error)
        self.info["confidence_error"] = current_error

        if self.split:
            children_error = 0
            for c in self.split.iter_subtrees():
                children_error += (c.info['num_samples']/n) * c.info["confidence_error"]

            if children_error > current_error:
                self.split = None
                self.info["splitted"] = True

    def get_target_distribution(self, sample):
        # TODO: descend into subtrees and return the leaf target distribution
        if self.split is not None:
            subtree = self.split(sample)
            if subtree is not None:
                return subtree.get_target_distribution(sample)
            else:
                return self.counts / self.info['num_samples']
        else:
            return self.counts / self.info['num_samples']
        
    def classify(self, sample):
        # TODO: classify the sample by descending into the appropriate subtrees.
        if self.split is not None:
            subtree = self.split(sample)
            if subtree is not None:
                return subtree.classify(sample)
            else:
                # idmax() == This method is the DataFrame version of ndarray.argmax
                return self.counts.idxmax()
        else:
            return self.counts.idxmax()
                
    def draw(self, print_info=True):
        dot = graphviz.Digraph()
        self.add_to_graphviz(dot, print_info)
        return dot

    def add_to_graphviz(self, dot, print_info):
        freqs = self.counts / self.counts.sum()
        freqs = dict(freqs)
        colors = []
        freqs_info = []
        for i, c in enumerate(self.all_targets):
            freq = freqs.get(c, 0.0)
            if freq > 0:
                colors.append(f"{i%9 + 1};{freq}")
                freqs_info.append(f'{c}:{freq:.2f}')
        colors = ':'.join(colors)
        labels = [' '.join(freqs_info)]
        if print_info:
            for k,v in self.info.items():
                labels.append(f'{k} = {v}')
        if self.split:
            labels.append(f'split by: {self.split.attr}')
        dot.node(f'{id(self)}',
                 label='\n'.join(labels), 
                 shape='box',
                 style='striped',
                 fillcolor=colors,
                 colorscheme='set19')
        if self.split:
            self.split.add_to_graphviz(dot, self, print_info)

class NumericalSplit(AbstractSplit):
    def __init__(self, attr, th):
        super(NumericalSplit, self).__init__(attr)
        self.th = th
    
    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = (
            Tree(df[df[self.attr] <= self.th], **subtree_kwargs),
            Tree(df[df[self.attr] > self.th], **subtree_kwargs))

    def __call__(self, x):
        if x[self.attr] <= self.th:
            return self.subtrees[0]
        else:
            return self.subtrees[1]
    
    def __str__(self):
        return f"NumericalSplit: {self.attr} <= {self.th}"

    def iter_subtrees(self):
        return self.subtrees
    
    def add_to_graphviz(self, dot, parent, print_info):
        self.subtrees[0].add_to_graphviz(dot, print_info)
        dot.edge(f'{id(parent)}', f'{id(self.subtrees[0])}',
                 label=f'<= {self.th:.2f}')
        self.subtrees[1].add_to_graphviz(dot, print_info)
        dot.edge(f'{id(parent)}', f'{id(self.subtrees[1])}',
                 label=f'> {self.th:.2f}')


def get_numrical_split_and_purity(df, parent_purity, purity_fun, attr,
                                  normalize_by_split_entropy=False):
    """Find best split thereshold and compute the average purity after a split.
    Args:
        df: a dataframe
        parent_purity: purity of the parent node
        purity_fun: function to compute the purity
        attr: attribute over whihc to split the dataframe
        normalize_by_split_entropy: if True, divide the purity gain by the split
            entropy (to compute https://en.wikipedia.org/wiki/Information_gain_ratio)
    
    Returns:
        pair of (split, purity_gain)
    """
    attr_df = df[[attr, 'target']].sort_values(attr)
    targets = attr_df['target']
    values = attr_df[attr]
    # Start with a split that puts all the samples into the right subtree
    right_counts = targets.value_counts()
    left_counts = right_counts * 0

    best_split = None
    best_purity_gain = -1
    N = len(attr_df)
    for row_i in range(N - 1):
        # Update the counts of targets in the left and right subtree and compute
        # the purity of the slipt for all possible thresholds!
        # Return the best split found.

        # Remember that the attribute may have duplicate values and all samples
        # with the same attribute value must end in the same subtree!
        row_target = targets.iloc[row_i]
        left_counts[row_target] -= 1
        right_counts[row_target] -= 1

        if attr_df.iloc[row_i][0] != attr_df.iloc[row_i + 1][0]:
            children_purity = (row_i + 1) * purity_fun(left_counts) + (N - row_i - 1) * purity_fun(right_counts)
            children_purity /= N
            purity = parent_purity - children_purity

            if purity > best_purity_gain:
                best_purity_gain = purity
                # our threshold
                attr_mean = (attr_df.iloc[row_i][0] + attr_df.iloc[row_i + 1][0]) / 2
                best_split = NumericalSplit(attr, attr_mean)

                
    if normalize_by_split_entropy:
        best_purity_gain /= entropy(targets.value_counts())
    return best_split, best_purity_gain

class RandomForest:
    def __init__(self, train, test, trees_num, criterion, nattrs):
        self.train = train
        self.test = test
        self.trees_num = trees_num
        self.criterion = criterion
        self.nattrs = nattrs
        self.trees = []
        self.errors = [] # array of tuples of 3 args
        self.make_forest()

    def make_forest(self):
        for t in range(self.trees_num):
            tree, oob = self.make_tree()
            self.trees.append(tree)
            print("Tree ",t," planted")

            # error counts
            tree_error = self.tree_error(tree, self.test)
            oob_error = self.tree_error(tree, oob)
            forest_error = self.forest_error(self.test)
            self.errors.append([tree_error, oob_error, forest_error])

    def make_tree(self):
        # with bagging
        train_df = resample(self.train, n_samples=len(self.train))
        # print("PPPOF1.5")
        # oob = [i for i in train_df if i not in self.train]
        oob = pd.DataFrame(self.train.loc[i] for i in self.train.index if i not in train_df.index)
        tree = Tree(train_df, criterion=self.criterion, nattrs=self.nattrs)
        # print(oob)
        return tree, oob

    def tree_error(self, tree, dataset):
        targets = [tree.classify(dataset.iloc[i]) for i in range(len(dataset))]
        # print(targets)
        # print(dataset['target'])
        classification = np.array(np.array(dataset['target']) == targets)
        return (len(classification) - np.count_nonzero(classification)) / len(dataset)

    def forest_error(self, dataset):
        forest_targets = np.array([[t.classify(dataset.iloc[i]) for i in range(len(dataset))] for t in self.trees])

        after_majority_voting = []
        for tests in range(len(dataset)):
            # print(forest_targets)
            trees_guess = forest_targets[:, tests]
            best_guess = sstats.mode(trees_guess)[0][0]
            # print(trees_guess)
            # print(best_guess)
            # assert 0 == 1
            after_majority_voting.append(best_guess)

        classification = np.array(dataset['target'] == np.array(after_majority_voting))
        return (len(classification) - np.count_nonzero(classification)) / len(dataset)

    def mean_tree_errors(self):
        return np.array(self.errors).mean(axis=0)

    def mean_agreement(self):
        forest_targets = np.array([[t.classify(self.test.iloc[i]) for i in range(len(self.test))] for t in self.trees])

        res = 0
        for i in range(len(self.trees)):
            for j in range(i+1, len(self.trees)):
                simmilar = [forest_targets[i] == forest_targets[j]]
                res += np.count_nonzero(simmilar) / len(self.test)
        
        if res == 0:
            return 0
        return res / (len(self.trees) * (len(self.trees)-1))/2
            

    def print_forest(self):
        for i in range(self.trees_num):
            print("Tree {}: RF Err rate {}\t Err rate {}\t OOB err rate {}".format(i,round(self.errors[i][2],3),round(self.errors[i][0],3), round(self.errors[i][1],3)))
