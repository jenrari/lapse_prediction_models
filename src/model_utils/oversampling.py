from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE


def over_sm(X_train, y_train):
    sm = SMOTE(sampling_strategy=0.2, k_neighbors=4,random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    return X_train_sm, y_train_sm


def over_adasyn(X_train, y_train):
    adasyn = ADASYN(sampling_strategy=0.2, n_neighbors=4,random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    return X_train_adasyn, y_train_adasyn


def over_bsm(X_train, y_train):
    bsm = BorderlineSMOTE(kind='borderline-1', sampling_strategy=0.2, k_neighbors=4, random_state=42)
    X_train_bsm, y_train_bsm = bsm.fit_resample(X_train, y_train)
    return X_train_bsm, y_train_bsm


def over_svmsm(X_train, y_train):
    svmsm = SVMSMOTE(sampling_strategy=0.2, k_neighbors=4, m_neighbors=5, random_state=42)
    X_train_svmsm, y_train_svmsm = svmsm.fit_resample(X_train, y_train)
    return X_train_svmsm, y_train_svmsm
