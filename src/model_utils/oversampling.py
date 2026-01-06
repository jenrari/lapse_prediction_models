from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler


def over_sm(X_train, y_train, sampling_strategy=0.2):
    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=3,random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    return X_train_sm, y_train_sm


def over_adasyn(X_train, y_train, sampling_strategy=0.2):
    adasyn = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=3,random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    return X_train_adasyn, y_train_adasyn


def over_bsm(X_train, y_train, sampling_strategy=0.2):
    bsm = BorderlineSMOTE(kind='borderline-1', sampling_strategy=sampling_strategy, k_neighbors=3, random_state=42)
    X_train_bsm, y_train_bsm = bsm.fit_resample(X_train, y_train)
    return X_train_bsm, y_train_bsm


def over_svmsm(X_train, y_train, sampling_strategy=0.2):
    svmsm = SVMSMOTE(sampling_strategy=sampling_strategy, k_neighbors=3, m_neighbors=10, random_state=42)
    X_train_svmsm, y_train_svmsm = svmsm.fit_resample(X_train, y_train)
    return X_train_svmsm, y_train_svmsm

def over_random(X_train, y_train, sampling_strategy=0.2):
    ro = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_ro, y_train_ro = ro.fit_resample(X_train, y_train)
    return X_train_ro, y_train_ro
