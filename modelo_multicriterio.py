import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

criterios = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
pesos = np.array([0.18, 0.22, 0.16, 0.14, 0.12, 0.18])   # ponderaciones; suman 1

tipologias = {
    'T1': ('O&M predictivo para parques solares y eólicos', 'Energía', [78, 60, 72, 80, 78, 75]),
    'T2': ('Microrredes con gestión inteligente de demanda', 'Energía/agua', [75, 65, 80, 58, 60, 60]),
    'T3': ('Desalinización renovable con gestión hídrica', 'Energía/agua', [70, 78, 82, 45, 50, 45]),
    'T4': ('Orquestador de gestión hídrica comunitaria', 'Energía/agua', [62, 88, 70, 70, 72, 68]),
    'T5': ('Torre de control logística con cadena de frío', 'Logística', [72, 62, 60, 62, 70, 66]),
    'T6': ('Consolidador inteligente de carga regional', 'Logística', [68, 60, 64, 60, 72, 62]),
    'T7': ('Pesca y acuicultura artesanal inteligente', 'Agro-pesca', [64, 82, 68, 55, 62, 55]),
    'T8': ('Agricultura de precisión para zonas áridas', 'Agro-pesca', [60, 70, 75, 52, 66, 54]),
    'T9': ('Capacidad de carga turística y destino inteligente', 'Turismo', [70, 72, 85, 60, 66, 64]),
    'T10': ('Revenue management para turismo comunitario', 'Turismo', [72, 75, 66, 64, 72, 70]),
    'T11': ('Guía cultural conversacional español/wayuunaiki', 'Turismo', [58, 80, 60, 66, 74, 64]),
    'T12': ('Clasificación de residuos y simbiosis industrial', 'Economía circular', [66, 66, 84, 56, 60, 56]),
    'T13': ('Valorización inteligente de residuos orgánicos', 'Economía circular', [60, 64, 80, 54, 58, 54]),
    'T14': ('Trazabilidad con blockchain para artesanía Wayuu', 'Manuf./artesanías', [70, 86, 64, 62, 70, 66]),
    'T15': ('Priorizador de inversión pública (multicriterio)', 'Servicios digitales', [58, 68, 64, 72, 70, 66]),
}

cods = list(tipologias)
nombres = {c: tipologias[c][0] for c in cods}
sectores = {c: tipologias[c][1] for c in cods}
M = np.array([tipologias[c][2] for c in cods], dtype=float)

total = M @ pesos
tabla = pd.DataFrame({'Codigo': cods,
                      'Tipologia': [nombres[c] for c in cods],
                      'Sector': [sectores[c] for c in cods],
                      'Total': np.round(total, 1)})
tabla = tabla.sort_values('Total', ascending=False).reset_index(drop=True)
tabla.insert(0, 'Orden', np.arange(1, len(tabla) + 1))
cv = 100 * total.std() / total.mean()
print('Orden de mérito:')
print(tabla.to_string(index=False))
print('\nCoeficiente de variación del puntaje total: %.1f%%' % cv)

impacto = (M[:, 0]*pesos[0] + M[:, 1]*pesos[1] + M[:, 2]*pesos[2]) / pesos[:3].sum()
viabilidad = (M[:, 3]*pesos[3] + M[:, 4]*pesos[4] + M[:, 5]*pesos[5]) / pesos[3:].sum()

def frontera_pareto(x, y):
    eficientes = []
    for i in range(len(x)):
        dominado = any((x[j] >= x[i]) and (y[j] >= y[i]) and ((x[j] > x[i]) or (y[j] > y[i]))
                       for j in range(len(x)) if j != i)
        if not dominado:
            eficientes.append(i)
    return eficientes

pf = frontera_pareto(impacto, viabilidad)
r_iv, p_iv = stats.pearsonr(impacto, viabilidad)
print('\nFrontera de Pareto:', sorted(cods[i] for i in pf))
print('Correlación de Pearson impacto-viabilidad: r = %.3f (p = %.3f, n = %d)' % (r_iv, p_iv, len(cods)))

def renormalizar(w):
    return w / w.sum()

escenarios = {
    'Base': pesos,
    'Optimista': renormalizar(pesos * np.array([1.2, 1, 1, 1, 1, 1.2])),
    'Restrictivo': renormalizar(pesos * np.array([1, 1.2, 1.2, 1, 1.2, 1])),
}
rangos = {nom: stats.rankdata(-(M @ w)).astype(int) for nom, w in escenarios.items()}
sensibilidad = pd.DataFrame(rangos, index=cods)
sensibilidad['Variacion_max'] = sensibilidad.max(axis=1) - sensibilidad.min(axis=1)
sensibilidad = sensibilidad.sort_values('Base')
print('\nAnálisis de sensibilidad (orden por escenario):')
print(sensibilidad.to_string())

semilla = 42
n_simulaciones = 30000
concentracion = 80

rng = np.random.default_rng(semilla)
pesos_sim = rng.dirichlet(pesos * concentracion, size=n_simulaciones)
puntajes_sim = pesos_sim @ M.T
ordenes = (-puntajes_sim).argsort(axis=1).argsort(axis=1) + 1

p_top1 = (ordenes == 1).mean(axis=0) * 100
p_top3 = (ordenes <= 3).mean(axis=0) * 100
p_top5 = (ordenes <= 5).mean(axis=0) * 100

montecarlo = pd.DataFrame({'Codigo': cods,
                           'Tipologia': [nombres[c] for c in cods],
                           'P_top1_%': np.round(p_top1, 1),
                           'P_top3_%': np.round(p_top3, 1),
                           'P_top5_%': np.round(p_top5, 1)})
montecarlo = montecarlo.sort_values('P_top3_%', ascending=False).reset_index(drop=True)

rho = np.array([stats.spearmanr(total, puntajes_sim[k]).correlation
                for k in range(0, n_simulaciones, 30)])
spearman_medio = rho.mean()
spearman_ic = (np.percentile(rho, 5), np.percentile(rho, 95))
print('\nRobustez de Monte Carlo (probabilidades, %):')
print(montecarlo.head(8).to_string(index=False))
print('\nSpearman (simulación vs. base): media = %.3f; IC 90%% = [%.3f, %.3f]' % (spearman_medio, spearman_ic[0], spearman_ic[1]))

en_frontera = set(pf)
fig, ax = plt.subplots(figsize=(7, 5))
for i, c in enumerate(cods):
    marca = i in en_frontera
    ax.scatter(impacto[i], viabilidad[i], s=90 if marca else 55,
               color='#E8A317' if marca else '#2E6B7E', edgecolor='white', zorder=3)
    ax.annotate(c, (impacto[i], viabilidad[i]), xytext=(5, 4),
                textcoords='offset points', fontsize=8)
orden_pf = sorted(pf, key=lambda i: impacto[i])
ax.plot([impacto[i] for i in orden_pf], [viabilidad[i] for i in orden_pf],
        '--', color='#E8A317', label='Frontera de Pareto')
ax.set_xlabel('Impacto'); ax.set_ylabel('Viabilidad')
ax.set_title('Frontera de Pareto del portafolio (impacto vs. viabilidad)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('frontera_pareto.png', dpi=200, bbox_inches='tight')

top = montecarlo.head(8)
x = np.arange(len(top)); ancho = 0.27
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.bar(x - ancho, top['P_top1_%'], ancho, label='P(primer lugar)', color='#E8A317')
ax.bar(x,         top['P_top3_%'], ancho, label='P(top 3)', color='#2E6B7E')
ax.bar(x + ancho, top['P_top5_%'], ancho, label='P(top 5)', color='#1F4E5F')
ax.set_xticks(x); ax.set_xticklabels(top['Codigo'])
ax.set_ylabel('Probabilidad (%)')
ax.set_title('Robustez del ordenamiento — 30 000 simulaciones de Monte Carlo')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('robustez_montecarlo.png', dpi=200, bbox_inches='tight')
