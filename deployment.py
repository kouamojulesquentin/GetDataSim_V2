import pandas as pd
import numpy as np
from scipy.fft import fft
import joblib

def deployment(new_data_path, model_path):
    """
    Fonction pour déployer le modèle sur de nouvelles données
    
    Parameters:
    -----------
    new_data_path : str
        Chemin vers le fichier CSV contenant les nouvelles données
    model_path : str
        Chemin vers le modèle sauvegardé (.joblib)
    
    Returns:
    --------
    predictions : array
        Les prédictions pour les nouvelles données
    probabilities : array
        Les probabilités associées à chaque prédiction (si disponible)
    """
    
    # Charger les nouvelles données
    new_data = pd.read_csv(new_data_path, header=0, index_col=0)
    
    # Charger le modèle
    model = joblib.load(model_path)
    
    # Vérifier si c'est un modèle temporel ou fréquentiel
    is_frequency_model = 'frequency' in model_path
    
    def time_to_frequency(v_out_str):
        N = len(v_out_str.split(','))
        v_out_list = list(map(float, v_out_str.split(',')))
        frequency_domain = fft(v_out_list)
        frequency_domain = np.abs(frequency_domain[:N//2+1])
        return frequency_domain
    
    def convert_v_out(value):
        suffixes = {'a': 1e-18, 'z': 1e-21, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'c': 1e-2, 'd': 1e-1, '': 1, 'da': 1e1, 'h': 1e2, 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18}
        parts = value.split(',')
        converted_parts = []
        for part in parts:
            if part[-1].isalpha():
                num, suffix = part[:-1], part[-1]
                try:
                    converted_parts.append(float(num) * suffixes[suffix])
                except ValueError:
                    converted_parts.append(float('nan'))
            else:
                try:
                    converted_parts.append(float(part))
                except ValueError:
                    converted_parts.append(float('nan'))
        return ','.join(map(str, converted_parts))

    def apply_value_conversion(data, output):
        df = data
        for col in df.columns:
            df[col] = df[col].apply(convert_v_out)
        return df
    
    new_data = apply_value_conversion(new_data, 'V_out')



    # Préparer les données selon le type de modèle
    if is_frequency_model:
        # Transformer les données en domaine fréquentiel
        X_new = new_data['V_out'].apply(time_to_frequency)
        X_new = X_new.apply(lambda x: list(x)).tolist()
        X_new = pd.DataFrame(X_new)
    else:
        X_new = new_data['V_out'].apply(lambda x: list(map(float, x.split(',')))).tolist()
        X_new = pd.DataFrame(X_new)
        X_new = new_data['V_out'].apply(lambda x: list(map(float, x.split(',')))).tolist()
    
    # Faire les prédictions
    predictions = model.predict(X_new)
    
    # Obtenir les probabilités si le modèle le permet
    try:
        probabilities = model.predict_proba(X_new)
    except AttributeError:
        probabilities = None
        print("Note: Ce modèle ne fournit pas de probabilités pour les prédictions")
    
    # Créer un DataFrame avec les résultats
    results = pd.DataFrame({
        'prediction': predictions
    })
    
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            results[f'probability_class_{i}'] = probabilities[:, i]
    
    # Sauvegarder les résultats
    results.to_csv('predictions_results.csv')
    print("Les prédictions ont été sauvegardées dans 'predictions_results.csv'")
    
    return predictions, probabilities
