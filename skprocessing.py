
import sklearn
import json

WRITEMODE = 'w'
FILENAME = 'OUT.json'

global config
config = {}

def process_NuSVC(modelfile):
    return process_ALL(modelfile)

def process_LinearSVR(modelfile):
    return process_ALL(modelfile)

def process_ALL(modelfile):
    config["skparams"] = modelfile.get_params()    
    if hasattr(modelfile, 'class_weight_'):
        config["class_weight_"] = modelfile.class_weight_.tolist() 
    if hasattr(modelfile, 'coef_'):
        config["coef_"] = modelfile.coef_.tolist()
    if hasattr(modelfile, 'dual_coef_'):
        config["dual_coef_"] = modelfile.dual_coef_.tolist().tolist()
    if hasattr(modelfile, 'intercept_'):
        config["intercept_"] = modelfile.intercept_.tolist()
    if hasattr(modelfile, 'classes_'):
        config["classes_"] = modelfile.classes_.tolist()
    if hasattr(modelfile, 'n_features_in_'):
        config["n_features_in_"] = modelfile.n_features_in_ 
    if hasattr(modelfile, 'feature_names_in_'):
        config["feature_names_in_"] = modelfile.feature_names_in_.tolist()
    if hasattr(modelfile, 'n_iter_'):
        config["n_iter_"] = modelfile.n_iter_.item()
    if hasattr(modelfile, 'fit_status_'):
        config["fit_status_"] = modelfile.fit_status_ 
    if hasattr(modelfile, 'support_'):
        config["support_"] = modelfile.support_.tolist()
    if hasattr(modelfile, 'support_vectors_'):
        config["support_vectors_"] = modelfile.support_vectors_.tolist()
    if hasattr(modelfile, 'n_support_'):
        config["n_support_"] = modelfile.n_support_.tolist()
    if hasattr(modelfile, 'probA_'):
        config["probA_"] = modelfile.probA_.tolist()
    if hasattr(modelfile, 'probB_'):
        config["probB_"] = modelfile.probB_.tolist()
    if hasattr(modelfile, 'shape_fit_'):
        config["shape_fit_"] = modelfile.shape_fit_ 

    return config

    
def process_LinearSVC(modelfile):
    config["skparams"] = modelfile.get_params()
    if hasattr(modelfile, 'coef_'):
        config["coef_"] = modelfile.coef_.tolist()
    if hasattr(modelfile, 'intercept_'):
        config["intercept_"] = modelfile.intercept_.tolist()
    if hasattr(modelfile, 'classes_'):
        config["classes_"] = modelfile.classes_.tolist()
    if hasattr(modelfile, 'n_features_in_'):
        config["n_features_in_"] = modelfile.n_features_in_ 
    if hasattr(modelfile, 'feature_names_in_'):
        config["feature_names_in_"] = modelfile.feature_names_in_.tolist()
    if hasattr(modelfile, 'n_iter_'):
        config["n_iter_"] = modelfile.n_iter_.item()

    return config

def process_SVC(modelfile):
    config["skparams"] = modelfile.get_params() 
    if hasattr(modelfile, 'class_weight_'):
        config["class_weight_"] = modelfile.class_weight_.tolist()
    if hasattr(modelfile, 'coef_'):
        config["coef_"] = modelfile.coef_.tolist() 
    if hasattr(modelfile, 'dual_coef_'):
        config["dual_coef_"] = modelfile.dual_coef_.tolist()
    if hasattr(modelfile, 'intercept_'):
        config["intercept_"] = modelfile.intercept_.tolist()
    if hasattr(modelfile, 'classes_'):
        config["classes_"] = modelfile.classes_.tolist()
    if hasattr(modelfile, 'n_features_in_'):
        config["n_features_in_"] = modelfile.n_features_in_ 
    if hasattr(modelfile, 'feature_names_in_'):
        config["feature_names_in_"] = modelfile.feature_names_in_.tolist()
    if hasattr(modelfile, 'fit_status_'):
        config["fit_status_"] = modelfile.fit_status_ 
    if hasattr(modelfile, 'support_'):
        config["support_"] = modelfile.support_.tolist()
    if hasattr(modelfile, 'support_vectors_'):
        config["support_vectors_"] = modelfile.support_vectors_.tolist()
    if hasattr(modelfile, 'n_support_'):
        config["n_support_"] = modelfile.n_support_.tolist()
    if hasattr(modelfile, 'probA_'):
        config["probA_"] = modelfile.probA_.tolist()
    if hasattr(modelfile, 'probB_'):
        config["probB_"] = modelfile.probB_.tolist()
    if hasattr(modelfile, 'shape_fit_'):
        config["shape_fit_"] = modelfile.shape_fit_ 

    return config

def process_NuSVR(modelfile):
    config["skparams"] = modelfile.get_params() 
    if hasattr(modelfile, 'class_weight_'):
        config["class_weight_"] = modelfile.class_weight_.tolist()
    if hasattr(modelfile, 'coef_'):
        config["coef_"] = modelfile.coef_.tolist()
    if hasattr(modelfile, 'dual_coef_'):
        config["dual_coef_"] = modelfile.dual_coef_.tolist()
    if hasattr(modelfile, 'intercept_'):
        config["intercept_"] = modelfile.intercept_.tolist()
    if hasattr(modelfile, 'n_features_in_'):
        config["n_features_in_"] = modelfile.n_features_in_ 
    if hasattr(modelfile, 'feature_names_in_'):
        config["feature_names_in_"] = modelfile.feature_names_in_.tolist()
    if hasattr(modelfile, 'fit_status_'):
        config["fit_status_"] = modelfile.fit_status_ 
    if hasattr(modelfile, 'support_'):
        config["support_"] = modelfile.support_.tolist()
    if hasattr(modelfile, 'support_vectors_'):
        config["support_vectors_"] = modelfile.support_vectors_.tolist()
    if hasattr(modelfile, 'n_support_'):
        config["n_support_"] = modelfile.n_support_.tolist()
    if hasattr(modelfile, 'shape_fit_'):
        config["shape_fit_"] = modelfile.shape_fit_ 

    return config

def process_SVR(modelfile):
    config["skparams"] = modelfile.get_params() 
    if hasattr(modelfile, 'class_weight_'):
        config["class_weight_"] = modelfile.class_weight_.tolist()
    if hasattr(modelfile, 'coef_'):
        config["coef_"] = modelfile.coef_.tolist()
    if hasattr(modelfile, 'dual_coef_'):
        config["dual_coef_"] = modelfile.dual_coef_.tolist()
    if hasattr(modelfile, 'intercept_'):
        config["intercept_"] = modelfile.intercept_.tolist()
    if hasattr(modelfile, 'n_features_in_'):
        config["n_features_in_"] = modelfile.n_features_in_ 
    if hasattr(modelfile, 'feature_names_in_'):
        config["feature_names_in_"] = modelfile.feature_names_in_.tolist()
    if hasattr(modelfile, 'fit_status_'):
        config["fit_status_"] = modelfile.fit_status_ 
    if hasattr(modelfile, 'support_'):
        config["support_"] = modelfile.support_.tolist()
    if hasattr(modelfile, 'support_vectors_'):
        config["support_vectors_"] = modelfile.support_vectors_.tolist()
    if hasattr(modelfile, 'n_support_'):
        config["n_support_"] = modelfile.n_support_.tolist()
    if hasattr(modelfile, 'shape_fit_'):
        config["shape_fit_"] = modelfile.shape_fit_ 

    return config
