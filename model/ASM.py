import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def generate_transactions(data, users):
    for i in range(5):
        # randomly generating the no of videos for perticular transaction
        no_of_v = np.random.randint(3, min(10, len(data)))

        # generating the transaction
        users.append(list(np.random.choice(np.arange(0, min(10, len(data))),
                     replace=False, size=no_of_v)))


def refine_frozenset(data):
    # Convert frozensets to lists using nested list comprehension
    arr = [[list(x) if isinstance(x, frozenset) else x for x in row]
           for row in data]

    # Concatenate each value of the frozensets with "video"
    for row in arr:
        for i in range(len(row)):
            if isinstance(row[i], list):
                row[i] = ["Video " + str(elem) for elem in row[i]]
            else:
                row[i] = str(row[i])

    return arr


def generate_association_rules(data):

    # creating five users with transactions to perform the association rule mining
    users = []
    generate_transactions(data, users)

    te = TransactionEncoder()
    te_ary = te.fit(users).transform(users)

    df = pd.DataFrame(te_ary, columns=te.columns_)

    # finding frequent patterns
    frequent_itemsets = apriori(df, min_support=0.7, use_colnames=True)

    # Generating association rules
    res = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=0.7)

    res1 = res[res["confidence"] >= 0.8]
    res1 = res1[["antecedents", "consequents", "confidence"]]

    c1 = df.columns
    c2 = frequent_itemsets.columns
    c3 = res1.columns
    data_transformation = df.to_numpy()
    fre_itemsets = frequent_itemsets.to_numpy()
    deduced_asso_rules = res1.to_numpy()

    fre_itemsets = refine_frozenset(fre_itemsets)
    deduced_asso_rules = refine_frozenset(deduced_asso_rules)

    return c1, c2, c3, data_transformation, fre_itemsets, deduced_asso_rules
