from sklearn.model_selection import train_test_split

def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)