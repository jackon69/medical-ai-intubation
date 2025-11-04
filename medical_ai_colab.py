# medical_ai_colab.py - VERSIONE COMPLETA PER COLAB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CLASSE DATABASE - DEFINITA PRIMA DI TUTTO
# =============================================================================
class MedicalDatabase:
    def __init__(self):
        self.train_file = 'train_dataset.csv'
        self.test_file = 'test_dataset.csv'
        self.new_data_file = 'new_cases.csv'
        
    def create_initial_datasets(self, n_samples=1000):
        """Crea dataset iniziali basati su distribuzioni realistiche"""
        np.random.seed(42)
        
        # Distribuzioni basate su letteratura medica
        data = {
            'eta': np.clip(np.random.normal(55, 15, n_samples), 18, 90).astype(int),
            'peso': np.clip(np.random.normal(75, 20, n_samples), 40, 150),
            'mallampati': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
            'stop_bang': np.random.choice(range(0, 9), n_samples, p=[0.1, 0.15, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04]),
            'al_ganzuri': np.clip(np.random.normal(4.5, 1.2, n_samples), 2, 8),
            'dimensioni': np.clip(np.random.normal(16, 2, n_samples), 10, 25),
            'dii': np.clip(np.random.normal(5.5, 1.5, n_samples), 2, 10),
            'cormack': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # TARGET BINARIO: Cormack > 2 = Intubazione difficile
        df['intubazione_difficile'] = (df['cormack'] > 2).astype(int)
        
        # Split train/test
        train_df = df.iloc[:800].copy()
        test_df = df.iloc[800:].copy()
        
        # Salva datasets
        train_df.to_csv(self.train_file, index=False)
        test_df.to_csv(self.test_file, index=False)
        
        # Crea file per nuovi casi (vuoto)
        pd.DataFrame(columns=train_df.columns).to_csv(self.new_data_file, index=False)
        
        print("✅ Database iniziali creati!")
        print(f"📊 Prevalenza intubazione difficile: {df['intubazione_difficile'].mean():.2%}")
        return train_df, test_df
    
    def add_new_case(self, features, cormack_observed):
        """Aggiunge un nuovo caso con il Cormack osservato"""
        try:
            # Carica nuovi casi esistenti
            if os.path.exists(self.new_data_file):
                new_cases = pd.read_csv(self.new_data_file)
            else:
                new_cases = pd.DataFrame(columns=['eta', 'peso', 'mallampati', 'stop_bang', 
                                                'al_ganzuri', 'dimensioni', 'dii', 'cormack', 
                                                'intubazione_difficile'])
            
            # Crea nuovo caso
            new_case = features.copy()
            new_case['cormack'] = cormack_observed
            new_case['intubazione_difficile'] = 1 if cormack_observed > 2 else 0
            
            # Aggiungi ai nuovi casi
            new_cases = pd.concat([new_cases, pd.DataFrame([new_case])], ignore_index=True)
            new_cases.to_csv(self.new_data_file, index=False)
            
            print("✅ Nuovo caso aggiunto al database!")
            return True
            
        except Exception as e:
            print(f"❌ Errore nell'aggiungere nuovo caso: {e}")
            return False
    
    def update_training_data(self):
        """Aggiorna il dataset di training con i nuovi casi"""
        try:
            # Carica dati attuali
            train_df = pd.read_csv(self.train_file)
            
            if os.path.exists(self.new_data_file):
                new_cases = pd.read_csv(self.new_data_file)
            else:
                new_cases = pd.DataFrame()
            
            if len(new_cases) > 0:
                # Combina con nuovi casi
                updated_train = pd.concat([train_df, new_cases], ignore_index=True)
                updated_train.to_csv(self.train_file, index=False)
                
                # Resetta nuovi casi
                pd.DataFrame(columns=new_cases.columns).to_csv(self.new_data_file, index=False)
                
                print(f"✅ Training data aggiornato! Nuovi casi aggiunti: {len(new_cases)}")
                return updated_train
            else:
                print("ℹ️  Nessun nuovo caso da aggiungere")
                return train_df
                
        except Exception as e:
            print(f"❌ Errore nell'aggiornamento: {e}")
            return None
