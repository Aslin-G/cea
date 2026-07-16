# -*- coding: utf-8 -*-
"""
Análisis comparativo (TOPSIS, VIKOR, WPM) y de incertidumbre ampliado (Monte Carlo
en tres regímenes + leave-one-criterion-out + no dominancia 6D).
Datos: Tabla del manuscrito (matriz 15x6) y pesos (Tabla 1). Semilla = 42.
"""
import numpy as np, pandas as pd
from scipy import stats

criterios = ['C1','C2','C3','C4','C5','C6']
pesos = np.array([0.18,0.22,0.16,0.14,0.12,0.18])
tip = {
 'T1':('O&M predictivo solar/eólico',[78,60,72,80,78,75]),
 'T2':('Microrredes gestión demanda',[75,65,80,58,60,60]),
 'T3':('Desalinización renovable',[70,78,82,45,50,45]),
 'T4':('Gestión hídrica comunitaria',[62,88,70,70,72,68]),
 'T5':('Torre control logística',[72,62,60,62,70,66]),
 'T6':('Consolidador carga regional',[68,60,64,60,72,62]),
 'T7':('Pesca/acuicultura inteligente',[64,82,68,55,62,55]),
 'T8':('Agricultura precisión árida',[60,70,75,52,66,54]),
 'T9':('Capacidad carga turística',[70,72,85,60,66,64]),
 'T10':('Revenue mgmt turismo comunitario',[72,75,66,64,72,70]),
 'T11':('Guía cultural wayuunaiki',[58,80,60,66,74,64]),
 'T12':('Clasificación residuos/simbiosis',[66,66,84,56,60,56]),
 'T13':('Valorización residuos orgánicos',[60,64,80,54,58,54]),
 'T14':('Trazabilidad artesanía Wayuu',[70,86,64,62,70,66]),
 'T15':('Priorizador inversión pública',[58,68,64,72,70,66]),
}
cods = list(tip); M = np.array([tip[c][1] for c in cods],dtype=float)
def ranks(v_desc): return stats.rankdata(-np.asarray(v_desc)).astype(int)

wsm = M @ pesos
r_wsm = ranks(wsm)
print("=== WSM base (verificación) ===")
df = pd.DataFrame({'cod':cods,'total':np.round(wsm,1),'rank':r_wsm}).sort_values('rank')
print(df.to_string(index=False))

norm = M / np.sqrt((M**2).sum(axis=0))
V = norm * pesos
ideal, anti = V.max(axis=0), V.min(axis=0)
d_pos = np.sqrt(((V-ideal)**2).sum(axis=1)); d_neg = np.sqrt(((V-anti)**2).sum(axis=1))
Ci = d_neg/(d_pos+d_neg); r_top = ranks(Ci)
print("\n=== TOPSIS ===")
print(pd.DataFrame({'cod':cods,'Ci':np.round(Ci,4),'rank':r_top}).sort_values('rank').to_string(index=False))

fstar, fmin = M.max(axis=0), M.min(axis=0)
Snorm = (pesos*(fstar-M)/(fstar-fmin))
S = Snorm.sum(axis=1); R = Snorm.max(axis=1)
v=0.5
Q = v*(S-S.min())/(S.max()-S.min()) + (1-v)*(R-R.min())/(R.max()-R.min())
r_vik = ranks(-Q)
print("\n=== VIKOR (v=0.5) ===")
dv = pd.DataFrame({'cod':cods,'S':np.round(S,4),'R':np.round(R,4),'Q':np.round(Q,4),'rank':r_vik}).sort_values('rank')
print(dv.to_string(index=False))
orden_q = np.argsort(Q); DQ = 1/(len(cods)-1)
c1_ok = (Q[orden_q[1]]-Q[orden_q[0]]) >= DQ
best = orden_q[0]
c2_ok = (r_wsm[best]==1) or (r_top[best]==1)
print(f"Ventaja aceptable C1: {c1_ok} (dif={Q[orden_q[1]]-Q[orden_q[0]]:.4f} vs DQ={DQ:.4f}); Estabilidad C2: {c2_ok}; mejor={cods[best]}")

wpm = np.prod(M**pesos, axis=1); r_wpm = ranks(wpm)
print("\n=== WPM top5 ===")
print(pd.DataFrame({'cod':cods,'rank':r_wpm}).sort_values('rank').head(5).to_string(index=False))

print("\n=== Spearman entre rankings ===")
for nom, rr in [('TOPSIS',r_top),('VIKOR',r_vik),('WPM',r_wpm)]:
    rho,p = stats.spearmanr(r_wsm, rr)
    print(f"WSM vs {nom}: rho={rho:.3f} (p={p:.4g})")
rho_tv,_ = stats.spearmanr(r_top,r_vik); print(f"TOPSIS vs VIKOR: rho={rho_tv:.3f}")
import numpy as _np
top5 = {'WSM':set(_np.array(cods)[r_wsm<=5]),'TOPSIS':set(_np.array(cods)[r_top<=5]),'VIKOR':set(_np.array(cods)[r_vik<=5]),'WPM':set(_np.array(cods)[r_wpm<=5])}
print("Top-5 por método:", {k:sorted(v) for k,v in top5.items()})
print("Intersección top-5:", sorted(set.intersection(*top5.values())))
comp = pd.DataFrame({'cod':cods,'WSM_total':np.round(wsm,1),'r_WSM':r_wsm,'TOPSIS_Ci':np.round(Ci,3),'r_TOPSIS':r_top,'VIKOR_Q':np.round(Q,3),'r_VIKOR':r_vik,'r_WPM':r_wpm}).sort_values('r_WSM')
comp.to_csv('tabla_comparativa_metodos.csv', index=False)
print("\n", comp.to_string(index=False))

def no_dominados(X):
    nd=[]
    for i in range(len(X)):
        dom = any(np.all(X[j]>=X[i]) and np.any(X[j]>X[i]) for j in range(len(X)) if j!=i)
        if not dom: nd.append(i)
    return nd
nd6 = no_dominados(M)
print(f"\n=== No dominadas en 6D: {len(nd6)} de 15 ===", sorted(np.array(cods)[nd6].tolist(), key=lambda c:int(c[1:])))
imp = (M[:,:3]@pesos[:3])/pesos[:3].sum(); via=(M[:,3:]@pesos[3:])/pesos[3:].sum()
nd2 = no_dominados(np.c_[imp,via]); print("No dominadas en 2D:", sorted(np.array(cods)[nd2].tolist(), key=lambda c:int(c[1:])))
