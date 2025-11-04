# 🎯 VERSIONE SEMPLICE - INPUT PREIMPOSTATI
def demo_semplice():
    ai_system = ColabMedicalAI()
    ai_system.initialize_system()
    
    # Menu semplice
    print("🎯 SCEGLI UN CASO DI TEST:")
    print("1. Paziente alto rischio (65 anni, Mallampati 3, STOP-BANG 6)")
    print("2. Paziente basso rischio (45 anni, Mallampati 1, STOP-BANG 2)")
    print("3. Paziente rischio moderato (55 anni, Mallampati 2, STOP-BANG 4)")
    
    scelta = input("\nScelta (1-3): ")
    
    casi = {
        '1': {'eta': 65, 'peso': 85, 'mallampati': 3, 'stop_bang': 6, 'al_ganzuri': 3.5, 'dimensioni': 18, 'dii': 4.5},
        '2': {'eta': 45, 'peso': 70, 'mallampati': 1, 'stop_bang': 2, 'al_ganzuri': 5.0, 'dimensioni': 15, 'dii': 6.0},
        '3': {'eta': 55, 'peso': 80, 'mallampati': 2, 'stop_bang': 4, 'al_ganzuri': 4.5, 'dimensioni': 16, 'dii': 5.5}
    }
    
    if scelta in casi:
        probabilita = ai_system.predict_case(casi[scelta])
        print(f"\n🎯 RISULTATO: {probabilita:.1%}")
        if probabilita > 0.5:
            print("⚠️  RISCHIO ALTO")
        else:
            print("✅ RISCHIO BASSO")
    else:
        print("❌ Scelta non valida
