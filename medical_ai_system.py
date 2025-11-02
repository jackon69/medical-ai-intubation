# medical_ai_system.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')

class DynamicMedicalAI:
    def __init__(self):
        self.db = MedicalDatabase()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['eta', 'peso', 'mallampati', 'stop_bang', 
                               'al_ganzuri', 'dimensioni', 'dii']
        
    def load_data(self):
        """Carica i dataset"""
        self.train_df = pd.read_csv('train_dataset.csv')
        self.test_df = pd.read_csv('test_dataset.csv')
        print("Dati caricati con successo!")
        
    def exploratory_analysis(self):
        """Analisi esplorativa completa"""
        print("\n=== ANALISI ESPLORATIVA ===")
        print(self.train_df.describe())
        
        # Distribuzioni
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(self.feature_columns + ['cormack']):
            self.train_df[col].hist(ax=axes[i], bins=20)
            axes[i].set_title(f'Distribuzione {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequenza')
        
        plt.tight_layout()
        plt.show()
        
        # Heatmap correlazione
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.train_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Mappa Correlazione - Variabili Mediche')
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analisi target
        target_dist = self.train_df['intubazione_difficile'].value_counts()
        print(f"\nDistribuzione Target:")
        print(f"Intubazione Facile (Cormack 1-2): {target_dist[0]} casi")
        print(f"Intubazione Difficile (Cormack 3-4): {target_dist[1]} casi")
        print(f"Prevalenza: {target_dist[1]/len(self.train_df):.2%}")
    
    def train_model(self, update_with_new_data=True):
        """Addestra il modello, opzionalmente con nuovi dati"""
        if update_with_new_data:
            self.db.update_training_data()
            self.load_data()  # Ricarica dati aggiornati
        
        # Preparazione dati
        X_train = self.train_df[self.feature_columns]
        y_train = self.train_df['intubazione_difficile']
        X_test = self.test_df[self.feature_columns]
        y_test = self.test_df['intubazione_difficile']
        
        # Standardizzazione
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Addestramento Random Forest
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Valutazione
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        print(f"\n=== PERFORMANCE MODELLO ===")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Curva ROC
        self.plot_roc_curve(X_test_scaled, y_test)
        
        # Salva modello
        joblib.dump(self.model, 'medical_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        return auc_score
    
    def plot_roc_curve(self, X_test, y_test):
        """Plot della curva ROC"""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Casuale')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificità)')
        plt.ylabel('True Positive Rate (Sensibilità)')
        plt.title('Curva ROC - Predizione Intubazione Difficile\n(Target: Cormack > 2)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Trova threshold ottimale
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\nThreshold ottimale (Youden): {optimal_threshold:.3f}")
        print(f"Sensibilità: {tpr[optimal_idx]:.3f}")
        print(f"Specificità: {1-fpr[optimal_idx]:.3f}")
    
    def predict_single_case(self, features):
        """Predice per un singolo caso"""
        if self.model is None:
            self.load_model()
        
        input_array = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        
        probability = self.model.predict_proba(input_scaled)[0, 1]
        prediction = self.model.predict(input_scaled)[0]
        
        return probability, prediction
    
    def load_model(self):
        """Carica modello pre-addestrato"""
        try:
            self.model = joblib.load('medical_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("Modello caricato con successo!")
        except:
            print("Modello non trovato, addestramento necessario...")
            self.train_model()
    
    def create_gui(self):
        """Crea interfaccia grafica completa"""
        root = tk.Tk()
        root.title("Sistema AI - Predizione Intubazione Difficile")
        root.geometry("500x700")
        
        # Stile
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Title.TLabel', background='#f0f0f0', font=('Arial', 14, 'bold'))
        style.configure('TButton', font=('Arial', 10))
        style.configure('Header.TFrame', background='#2c3e50')
        
        # Header
        header_frame = ttk.Frame(root, style='Header.TFrame')
        header_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(header_frame, text="PREDIZIONE INTUBAZIONE DIFFICILE", 
                 style='Title.TLabel', foreground='white', background='#2c3e50').pack(pady=10)
        
        # Main container
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Variabili per input
        entries = {}
        cormack_observed = tk.StringVar()
        
        # Form input paziente
        input_frame = ttk.LabelFrame(main_frame, text="Parametri Paziente")
        input_frame.pack(fill='x', pady=10)
        
        fields = [
            ('eta', 'Età (anni):', '18-90'),
            ('peso', 'Peso (kg):', '40-150'),
            ('mallampati', 'Mallampati (1-4):', '1=Facile, 4=Difficile'),
            ('stop_bang', 'STOP-BANG (0-8):', '0=Basso rischio, 8=Alto rischio'),
            ('al_ganzuri', 'Al-Ganzuri (cm):', 'Distanza mento-ioide'),
            ('dimensioni', 'Dimensioni (cm):', 'Circonferenza collo'),
            ('dii', 'DII (cm):', 'Distanza inter-incisivi')
        ]
        
        for field, label, hint in fields:
            frame = ttk.Frame(input_frame)
            frame.pack(fill='x', padx=10, pady=5)
            
            ttk.Label(frame, text=label, width=20).pack(side='left')
            entry = ttk.Entry(frame)
            entry.pack(side='left', fill='x', expand=True)
            ttk.Label(frame, text=hint, font=('Arial', 8), foreground='gray').pack(side='right')
            entries[field] = entry
        
        # Risultato predizione
        result_frame = ttk.LabelFrame(main_frame, text="Risultato Predizione")
        result_frame.pack(fill='x', pady=10)
        
        result_text = tk.Text(result_frame, height=4, width=60, font=('Arial', 10))
        result_text.pack(padx=10, pady=10, fill='x')
        
        # Input Cormack osservato
        cormack_frame = ttk.LabelFrame(main_frame, text="Conferma Cormack Osservato (Post-Intubazione)")
        cormack_frame.pack(fill='x', pady=10)
        
        ttk.Label(cormack_frame, text="Cormack osservato:").pack(side='left', padx=10, pady=10)
        cormack_combo = ttk.Combobox(cormack_frame, textvariable=cormack_observed, 
                                   values=['1', '2', '3', '4'], state='readonly')
        cormack_combo.pack(side='left', padx=10, pady=10)
        cormack_combo.set('1')
        
        # Pulsanti
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=20)
        
        def predict_and_display():
            """Esegue predizione e mostra risultati"""
            try:
                # Raccolta dati
                features = {}
                for field in self.feature_columns:
                    value = float(entries[field].get())
                    features[field] = value
                
                # Predizione
                probability, prediction = self.predict_single_case(features)
                
                # Display risultati
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"PROBABILITÀ INTUBAZIONE DIFFICILE: {probability:.1%}\n\n")
                
                if probability > 0.5:
                    result_text.insert(tk.END, "🎯 RISCHIO: ALTO\n", "high_risk")
                    result_text.insert(tk.END, "Raccomandazione: Preparare strumentazione per intubazione difficile")
                else:
                    result_text.insert(tk.END, "✅ RISCHIO: BASSO\n", "low_risk")
                    result_text.insert(tk.END, "Raccomandazione: Procedura standard")
                
                # Configura colori
                result_text.tag_configure("high_risk", foreground="red", font=('Arial', 11, 'bold'))
                result_text.tag_configure("low_risk", foreground="green", font=('Arial', 11, 'bold'))
                
            except Exception as e:
                messagebox.showerror("Errore", f"Inserisci valori validi!\n{str(e)}")
        
        def save_case():
            """Salva il caso con Cormack osservato"""
            try:
                # Raccolta dati
                features = {}
                for field in self.feature_columns:
                    value = float(entries[field].get())
                    features[field] = value
                
                cormack_val = int(cormack_observed.get())
                
                # Salva nel database
                success = self.db.add_new_case(features, cormack_val)
                
                if success:
                    messagebox.showinfo("Successo", "Caso salvato nel database!\nIl modello verrà aggiornato al prossimo training.")
                else:
                    messagebox.showerror("Errore", "Errore nel salvataggio del caso")
                    
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel salvataggio: {str(e)}")
        
        def retrain_model():
            """Riaddestra il modello con tutti i dati"""
            auc_score = self.train_model(update_with_new_data=True)
            messagebox.showinfo("Aggiornamento", f"Modello riaddestrato!\nAUC Score: {auc_score:.4f}")
        
        # Pulsanti
        ttk.Button(button_frame, text="PREDICI RISCHIO", 
                  command=predict_and_display).pack(side='left', padx=5)
        ttk.Button(button_frame, text="SALVA CASO", 
                  command=save_case).pack(side='left', padx=5)
        ttk.Button(button_frame, text="AGGIORNA MODELLO", 
                  command=retrain_model).pack(side='left', padx=5)
        
        root.mainloop()

# Esecuzione principale
if __name__ == "__main__":
    print("=== SISTEMA AI PER PREDIZIONE INTUBAZIONE DIFFICILE ===")
    print("Target: Cormack > 2 = Intubazione difficile")
    
    ai_system = DynamicMedicalAI()
    
    # Carica dati iniziali
    ai_system.load_data()
    
    # Analisi esplorativa
    ai_system.exploratory_analysis()
    
    # Addestra modello iniziale
    auc_score = ai_system.train_model(update_with_new_data=False)
    
    if auc_score >= 0.95:
        print("\n✅ Sistema pronto - AUC > 95% raggiunto!")
    else:
        print(f"\n⚠️  AUC: {auc_score:.4f} - Sistema comunque operativo")
    
    # Avvia interfaccia
    print("\nAvvio interfaccia grafica...")
    ai_system.create_gui()
