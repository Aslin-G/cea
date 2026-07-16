import numpy as np, pandas as pd
from scipy import stats
pesos = np.array([0.18,0.22,0.16,0.14,0.12,0.18])
tip = {'T1':[78,60,72,80,78,75],'T2':[75,65,80,58,60,60],'T3':[70,78,82,45,50,45],'T4':[62,88,70,70,72,68],
 'T5':[72,62,60,62,70,66],'T6':[68,60,64,60,72,62],'T7':[64,82,68,55,62,55],'T8':[60,70,75,52,66,54],
 'T9':[70,72,85,60,66,64],'T10':[72,75,66,64,72,70],'T11':[58,80,60,66,74,64],'T12':[66,66,84,56,60,56],
 'T13':[60,64,80,54,58,54],'T14':[70,86,64,62,70,66],'T15':[58,68,64,72,70,66]}
cods=list(tip); M=np.array(list(tip.values()),dtype=float)
wsm = M@pesos
Nsim=30000; seed=42; conc=80

def resumen(sc, tag):
    o = (-sc).argsort(axis=1).argsort(axis=1)+1
    p1=(o==1).mean(axis=0)*100; p3=(o<=3).mean(axis=0)*100; p5=(o<=5).mean(axis=0)*100
    idx=range(0,Nsim,30)
    rho=np.array([stats.spearmanr(wsm, sc[k]).correlation for k in idx])
    dd=pd.DataFrame({'cod':cods,'P_top1':np.round(p1,1),'P_top3':np.round(p3,1),'P_top5':np.round(p5,1)}).sort_values('P_top3',ascending=False)
    print(f"\n--- {tag} ---"); print(dd.head(8).to_string(index=False))
    print(f"Spearman medio={rho.mean():.3f}; IC90=[{np.percentile(rho,5):.3f},{np.percentile(rho,95):.3f}]")
    return o, dd

rng=np.random.default_rng(seed)
W=rng.dirichlet(pesos*conc,size=Nsim)
oA,ddA=resumen(W@M.T,"A: solo pesos")

rng=np.random.default_rng(seed)
E=rng.uniform(-10,10,size=(Nsim,15,6))
oB,ddB=resumen(np.einsum('kij,j->ki',np.clip(M[None]+E,0,100),pesos),"B: solo puntajes U[-10,10]")

rng=np.random.default_rng(seed)
W2=rng.dirichlet(pesos*conc,size=Nsim); E2=rng.uniform(-10,10,size=(Nsim,15,6))
scC=np.einsum('kij,kj->ki',np.clip(M[None]+E2,0,100),W2)
oC,ddC=resumen(scC,"C: conjunto")

np.save('ordenes_conjunto.npy', oC)
out=pd.DataFrame({'cod':cods})
for tag,dd in [('A',ddA),('B',ddB),('C',ddC)]:
    m=dd.set_index('cod')
    for col in ['P_top1','P_top3','P_top5']: out[f'{tag}_{col}']=out['cod'].map(m[col])
out.to_csv('montecarlo_regimenes.csv',index=False)

# LOCO
def ranks(v): return stats.rankdata(-np.asarray(v)).astype(int)
r_wsm=ranks(wsm); res={}
for j,cj in enumerate(['C1','C2','C3','C4','C5','C6']):
    w2=np.delete(pesos,j); w2/=w2.sum()
    res[f'sin_{cj}']=ranks(np.delete(M,j,axis=1)@w2)
loco=pd.DataFrame(res,index=cods); loco.insert(0,'base',r_wsm)
print("\n=== LOCO (top-7 base) ===")
print(loco[loco['base']<=7].sort_values('base').to_string())
d=loco.drop(columns='base')
shift=(d.sub(loco['base'],axis=0)).abs().max(axis=1)
print("Desplazamiento máx de rango por tipología (top7):"); print(shift[loco['base']<=7].sort_values(ascending=False).to_string())
loco.to_csv('loco.csv')
