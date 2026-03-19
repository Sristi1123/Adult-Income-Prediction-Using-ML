import shap
import lime
import lime.lime_tabular

def explain_predictions_shap(model, X_train, X_test, feature_names):
    """Generate SHAP explanations"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])
    return shap_values

def explain_predictions_lime(model, X_train, feature_names, class_names):
    """Generate LIME explanations"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=class_names
    )
    return explainer