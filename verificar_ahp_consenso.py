# -*- coding: utf-8 -*-
"""
Reproduce la derivación de las ponderaciones del manuscrito (Tabla 3 y sección 3.3):
matriz de comparaciones por pares del consenso del panel (escala de Saaty),
autovector principal, lambda_max, CI y CR (RI = 1,24 para n = 6).
Uso: python3 verificar_ahp_consenso.py
"""
import numpy as np

CRITERIOS = ['C1 Económico-productivo', 'C2 Social y transición justa', 'C3 Ambiental',
             'C4 Técnico', 'C5 Regulatorio-licenciamiento', 'C6 Financiero y de riesgo']
# Matriz de consenso (Tabla 3 del manuscrito). Recíproca por construcción.
A = np.array([
    [1,   1/2, 1,   2,   2,   1  ],
    [2,   1,   1,   2,   2,   1  ],
    [1,   1,   1,   2,   1,   1/2],
    [1/2, 1/2, 1/2, 1,   2,   1  ],
    [1/2, 1/2, 1,   1/2, 1,   1  ],
    [1,   1,   2,   1,   1,   1  ],
], dtype=float)
W_ADOPTADO = np.array([0.18, 0.22, 0.16, 0.14, 0.12, 0.18])  # Tabla 1
RI = 1.24

assert np.allclose(A * A.T, np.ones((6, 6))), "la matriz no es recíproca"
val, vec = np.linalg.eig(A)
k = np.argmax(val.real)
w = np.abs(vec[:, k].real); w = w / w.sum()
lmax = val[k].real
CI = (lmax - 6) / 5
CR = CI / RI

print("Autovector principal (prioridades):")
for c, wi in zip(CRITERIOS, w):
    print(f"  {c:<32} {wi:.3f}")
print(f"\nlambda_max = {lmax:.3f}   CI = {CI:.3f}   CR = {CR:.3f}  ({'consistente: CR < 0,10' if CR < 0.10 else 'INCONSISTENTE'})")
print(f"Desviación máxima frente al vector adoptado (Tabla 1): {np.max(np.abs(w - W_ADOPTADO)):.3f}")
assert CR < 0.10
assert np.max(np.abs(w - W_ADOPTADO)) < 0.01
orden = np.argsort(-w)
print("Ordenamiento del autovector:", ' > '.join(CRITERIOS[i].split()[0] for i in orden))
print("\nVerificación superada: los valores coinciden con la Tabla 3 y la sección 3.3 del manuscrito.")
