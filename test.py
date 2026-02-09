import numpy as np
from stable_baselines3 import PPO  # ✅ Ora attivo

# --- CARICAMENTO MODELLO ---
try:
    model = PPO.load("ppo_sailing_marl")
    print("✅ Modello 'ppo_sailing_marl' caricato correttamente!")
except Exception as e:
    print(f"❌ Errore nel caricamento del modello: {e}")
    print("Assicurati di aver fatto model.save('ppo_sailing_marl') prima di eseguire questo script.")
    exit()

# 1. Setup Statistiche
num_episodes = 100

counts = {
    "Vittorie boat_0": 0,
    "Vittorie boat_1": 0,
    "Collisioni": 0,
    "Fuori Campo": 0,
    "Timeout (250 step)": 0
}

# --- NUOVO: Struttura dati espansa per tracciare vittorie per posizione ---
position_stats = {
    "boat_0": {"Inside": 0, "Outside": 0, "Win_Inside": 0, "Win_Outside": 0},
    "boat_1": {"Inside": 0, "Outside": 0, "Win_Inside": 0, "Win_Outside": 0}
}
# -------------------------------------------------------------------------

metrics = {
    "boat_0": {"vmg": [], "polar": [], "path": []},
    "boat_1": {"vmg": [], "polar": [], "path": []}
}

print(f"🚀 Inizio validazione modello su {num_episodes} episodi...")

for i in range(num_episodes):
    observations, infos = env.reset()

    # --- Rilevamento Posizione Iniziale ---
    current_inside_agent = None # Variabile temporanea per questo episodio

    for a in env.possible_agents:
        if env.boat_states[a]['is_inside']:
            position_stats[a]["Inside"] += 1
            current_inside_agent = a # Memorizziamo chi è interno
        else:
            position_stats[a]["Outside"] += 1
    # ---------------------------------------

    terminated = False
    truncated = False
    last_infos = infos

    while not (terminated or truncated):
        actions = {}

        # --- LOGICA DI PREVISIONE PPO ---
        for agent_id in env.agents:
            obs = observations[agent_id]
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
            actions[agent_id] = action
        # --------------------------------

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if infos:
            last_infos = infos

        terminated = all(terminations.values())
        truncated = all(truncations.values())

    # --- ASSEGNAZIONE ESITO E STATISTICHE TATTICHE ---
    if env.winner:
        # Conteggio globale
        if env.winner == "boat_0": counts["Vittorie boat_0"] += 1
        else: counts["Vittorie boat_1"] += 1

        # --- NUOVO: Controllo tattico (Ha vinto partendo da dentro o fuori?) ---
        if env.winner == current_inside_agent:
            # Il vincitore era quello partito INTERNO
            position_stats[env.winner]["Win_Inside"] += 1
        else:
            # Il vincitore era quello partito ESTERNO
            position_stats[env.winner]["Win_Outside"] += 1
        # -----------------------------------------------------------------------

    else:
        if any(truncations.values()):
            counts["Timeout (250 step)"] += 1
        else:
            p0 = np.array([env.boat_states["boat_0"]['x'], env.boat_states["boat_0"]['y']])
            p1 = np.array([env.boat_states["boat_1"]['x'], env.boat_states["boat_1"]['y']])
            if np.linalg.norm(p0 - p1) < (env.boat_radius * 2.1):
                counts["Collisioni"] += 1
            else:
                counts["Fuori Campo"] += 1

    # --- RACCOLTA METRICHE ---
    for a in env.possible_agents:
        agent_info = last_infos.get(a, {})
        metrics[a]["vmg"].append(agent_info.get("avg_vmg", 0))
        metrics[a]["polar"].append(agent_info.get("polar_efficiency", 0))
        metrics[a]["path"].append(agent_info.get("path_efficiency", 0))

    if (i + 1) % 10 == 0:
        print(f"✅ Completati {i + 1} episodi...")

# --- REPORT FINALE ---
print("\n" + "="*80)
print(f"{'REPORT VALIDAZIONE PPO':^80}")
print("="*80)

print(f"{'Esito Gara':<25} | {'Frequenza':<10}")
print("-" * 80)
for esito, conteggio in counts.items():
    icona = "🟢" if "Vittorie" in esito else "🔴"
    print(f"{icona} {esito:<22} | {conteggio:<10}")

print("-" * 80)
# --- NUOVA TABELLA CON VITTORIE SPECIFICHE ---
print(f"{'Agente':<10} | {'Start IN':<10} | {'Vittorie (da IN)':<18} | {'Start OUT':<10} | {'Vittorie (da OUT)':<18}")
print("-" * 80)
for a in env.possible_agents:
    s_in = position_stats[a]["Inside"]
    w_in = position_stats[a]["Win_Inside"]
    s_out = position_stats[a]["Outside"]
    w_out = position_stats[a]["Win_Outside"]

    # Calcolo percentuali di conversione (Vittorie / Start)
    perc_win_in = (w_in / s_in * 100) if s_in > 0 else 0.0
    perc_win_out = (w_out / s_out * 100) if s_out > 0 else 0.0

    # Formattazione stringa con percentuale tra parentesi
    str_w_in = f"{w_in} ({perc_win_in:.0f}%)"
    str_w_out = f"{w_out} ({perc_win_out:.0f}%)"

    print(f"{a:<10} | {s_in:<10} | {str_w_in:<18} | {s_out:<10} | {str_w_out:<18}")

print("-" * 80)
print(f"{'Agente':<10} | {'VMG Medio':<12} | {'Eff. Polare':<12} | {'Eff. Rotta':<12}")
print("-" * 80)
for a in env.possible_agents:
    avg_vmg = np.mean(metrics[a]["vmg"])
    avg_polar = np.mean(metrics[a]["polar"]) * 100
    avg_path = np.mean(metrics[a]["path"]) * 100
    print(f"{a:<10} | {avg_vmg:<12.2f} | {avg_polar:>10.1f}% | {avg_path:>10.1f}%")

print("="*80)
success_rate = ((counts["Vittorie boat_0"] + counts["Vittorie boat_1"]) / num_episodes) * 100
print(f"🎯 Percentuale Successo Totale: {success_rate:.1f}%")
print("="*80 + "\n")